import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import geopy
from geopy import distance
import geographiclib
from geographiclib.geodesic import Geodesic
import datetime
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import uuid


class PeakPacer:

    def __init__(self, app):
        self.data_sampled = None
        self.data_split = None

        self.app = app

        @app.callback(
            Output(
                'map-plot',
                'figure',
                allow_duplicate=True),
            Input('main-plot', 'hoverData'),
            prevent_initial_call=True
        )
        def update_map(hoverData):
            if hoverData is None or self.data_sampled is None:
                return self.map_fig

            data = self.data_sampled

            updated_map_fig = self.map_fig
            index = hoverData["points"][0]["pointIndex"]
            updated_map_fig.data[1].lat = [data["latitude"].values[index]]
            updated_map_fig.data[1].lon = [data["longitude"].values[index]]

            updated_map_fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=data["latitude"].values[index],
                        lon=data["longitude"].values[index]),
                ))
            text = "Power: {} W <br> Speed: {} km/h <br> Wind: {} km/h <br> Slope: {}".format(
                np.around(data["power"].values[index]), np.around(data["speed"].values[index]), np.around(data["wind"].values[index]), np.around(data["slope"].values[index]))
            updated_map_fig.data[1].text = text
            return updated_map_fig

    def aero_force(self, velocity, air_density, cda, wind_speed):
        return 0.5 * cda * air_density * (velocity + wind_speed)**2

    def rolling_resistance(self, rolling_friction, masse, slope):
        return rolling_friction * masse * 9.8067 * \
            np.cos(np.arctan(slope * 0.01))

    def gravity(self, masse, slope):
        return masse * 9.8067 * np.sin(np.arctan(slope * 0.01))

    def propulsive_power(self, velocity, air_density, cda, wind_speed,
                         rolling_friction, masse, slope, efficiency, **kwargs):
        return velocity * efficiency * (aero_force(velocity, air_density, cda, wind_speed) +
                                        rolling_resistance(rolling_friction, masse, slope) + gravity(masse, slope))

    def velocity_from_power(self, power, data):

        order_3 = np.ones(len(power)) * self.rider_profil["efficiency"] * \
            0.5 * self.road_profil["air_density"] * self.rider_profil["cda"]
        order_2 = np.ones(len(power)) * self.rider_profil["efficiency"] * \
            self.road_profil["air_density"] * \
            self.rider_profil["cda"] * data["wind"].values
        order_1 = self.rider_profil["efficiency"] * (0.5 * self.road_profil["air_density"] * self.rider_profil["cda"] * data["wind"].values +
                                                     self.rolling_resistance(self.road_profil["rolling_friction"], self.rider_profil["masse"], data["slope"].values) + self.gravity(self.rider_profil["masse"], data["slope"].values))
        order_0 = -power

        coefficients = np.vstack([order_3, order_2, order_1, order_0]).T
        roots = [np.roots(coeff) for coeff in coefficients]
        velocities = np.array(
            [np.max(r[np.isreal(r) & (r >= 0)]).real for r in roots])
        return velocities

    def moving_average(self, a, n=1):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def target_power(self, x):

        def func(x, a):
            return (self.rider_profil["ftp"]) + a / x

        fatigue = 1

        popt, pcov = scipy.optimize.curve_fit(func, [3600, 300, 60],
                                              [self.rider_profil["ftp"], self.rider_profil["pma"] * fatigue, 1.8 * self.rider_profil["pma"] * fatigue])
        return int(func(x, *popt))

    def px2pt(self, power):
        t = np.cumsum(
            self.road_profil["dist_precision"] /
            self.velocity_from_power(
                power,
                self.data_sampled))
        x = np.arange(0, t[-1], self.road_profil["time_precision"])
        p_resamp = np.interp(x, t, power)
        return p_resamp

    def np_power(self, power):
        power = self.moving_average(
            power, 30 // self.road_profil["time_precision"])
        return np.mean(power**4)**(1 / 4)

    def travel_time(self, power):
        t = np.sum(
            self.data_split["dx"].values / self.velocity_from_power(power, self.data_split))
        return t

    def mean_power_constraint(self, power):
        t = self.data_split["dx"].values / \
            self.velocity_from_power(power, self.data_split)
        return self.target_power(np.sum(t)) - np.sum(t * power) / np.sum(t)

    def pma_power_constraint(self, power):
        t = self.data_split["dx"].values / \
            self.velocity_from_power(power, self.data_split)

        p_time = []
        for i, j in enumerate(t):
            p_time.extend([power[i]] * int(j / 10))
        return 0.95 * self.rider_profil["pma"] - \
            np.max(self.moving_average(p_time, 30))

    def minimize_time(self):
        start = np.random.normal(
            self.rider_profil["ftp"], 50, len(self.data_split["dx"].values))
        mean_power = ({'type': 'ineq', 'fun': self.mean_power_constraint})
        pma_power = ({'type': 'ineq', 'fun': self.pma_power_constraint})

        A = scipy.optimize.minimize(
            self.travel_time,
            start,
            options={
                "eps": 2e-2,
                "maxiter": 500,
                "ftol": 1e-3},
            method="SLSQP",
            bounds=[
                (1,
                 1000)],
            constraints=[
                mean_power,
                pma_power])
        print(A)
        return A.x, A.fun

    def wind(self, orientation, wind_direction, wind_force):
        wind_direction = np.ones_like(orientation) * wind_direction
        return wind_force * np.cos(np.deg2rad(orientation - wind_direction))

    def compute_split_average(self, values, splits):
        means = []
        for i, j in enumerate(splits[:-1]):
            means.append(np.mean(values[j:splits[i + 1]]))
        return np.asarray(means)

    def compute_split_sum(self, values, splits):
        means = []
        for i, j in enumerate(splits[:-1]):
            means.append(np.sum(values[j:splits[i + 1]]))
        return np.asarray(means)

    def set_parameters(self, rider_profil, road_profil):
        self.rider_profil = dict(rider_profil)
        self.road_profil = dict(road_profil)
        self.road_profil["dist_precision"] = 10
        self.road_profil["time_precision"] = 10

    def analysis(self):

        # Compute data from GPS
        geod = Geodesic.WGS84
        coords = self.road_profil["coordinates"]
        self.road_profil["distance"] = np.cumsum(
            [0] + [geod.Inverse(*j, *coords[i + 1])['s12'] for i, j in enumerate(coords[:-1])])
        self.road_profil["direction"] = [
            0] + [geod.Inverse(*j, *coords[i + 1])['azi1'] for i, j in enumerate(coords[:-1])]

        # Resample data
        x = np.arange(0, self.road_profil["distance"][-1],
                      self.road_profil["dist_precision"])
        elevation_sampled = np.interp(
            x, self.road_profil["distance"], self.road_profil["elevation"])
        direction_sampled = np.interp(
            x, self.road_profil["distance"], self.road_profil["direction"])
        slope_sampled = np.diff(elevation_sampled) / np.diff(x) * 100
        slope_sampled = np.insert(slope_sampled, 0, slope_sampled[0])
        latitude_sampled = np.interp(
            x, self.road_profil["distance"], [i for i, _ in coords])
        longitude_sampled = np.interp(
            x, self.road_profil["distance"], [i for _, i in coords])
        wind_sampled = self.wind(
            direction_sampled,
            np.ones_like(direction_sampled) *
            self.road_profil["wind_direction_param"],
            self.road_profil["wind_speed_param"])

        self.data_sampled = pd.DataFrame({"distance": x,
                                          "elevation": elevation_sampled,
                                          "direction": direction_sampled,
                                          "slope": slope_sampled,
                                          "latitude": latitude_sampled,
                                          "longitude": longitude_sampled,
                                          "wind": wind_sampled})

        # Detect split
        splits = np.sign(np.diff(scipy.signal.wiener(elevation_sampled, 5)))
        splits[splits == 0] = 1
        idx = np.where(splits[:-1] != splits[1:])[0]
        idx[0] = 0
        direction_change = np.abs(
            (np.diff(direction_sampled) + 180) %
            360 - 180)
        peaks, _ = scipy.signal.find_peaks(direction_change, height=40)
        idx = np.unique(np.sort(np.append(idx, peaks)))
        idx = idx[np.where(np.diff(idx) > 5)[0]]
        idx = np.append(idx, len(x) - 1)

        # Compute data mean on splits
        distance = x[idx]
        slope_split = self.compute_split_average(
            self.data_sampled["slope"].values, idx)
        wind_split = self.compute_split_average(
            self.data_sampled["wind"].values, idx)

        self.data_split = pd.DataFrame({"dx": np.diff(distance),
                                        "slope": slope_split,
                                        "wind": wind_split})

        # Minimize
        p, t = self.minimize_time()

        p_sampled = np.zeros_like(x)
        for i, j in enumerate(x):
            bound = np.argmax(j < distance)
            p_sampled[i] = p[bound - 1]
        self.data_sampled["power"] = p_sampled

        self.data_sampled["speed"] = self.velocity_from_power(
            p_sampled,
            self.data_sampled)

        time_split = np.diff(
            self.data_sampled["distance"].values) / self.data_sampled["speed"].values[:-1]
        time_exact = np.sum(np.diff(
            self.data_sampled["distance"].values) / self.data_sampled["speed"].values[:-1])
        p_t = self.px2pt(p_sampled)

        summary = "Time: " + str(datetime.timedelta(seconds=time_exact))
        summary += "\n"
        summary += "Speed: " + \
            str(np.around(distance[-1] / time_exact * 3.6)) + " km/h"
        summary += "\n"
        summary += "1 min: " + \
            str(np.around(np.max(self.moving_average(
                p_t, 60 // self.road_profil["time_precision"])))) + " W"
        summary += "\n"
        summary += "5 min: " + \
            str(np.around(np.max(self.moving_average(
                p_t, 300 // self.road_profil["time_precision"])))) + " W"
        summary += "\n"
        summary += "8 min: " + \
            str(np.around(np.max(self.moving_average(
                p_t, 480 // self.road_profil["time_precision"])))) + " W"
        summary += "\n"
        summary += "10 min: " + \
            str(np.around(np.max(self.moving_average(
                p_t, 600 // self.road_profil["time_precision"])))) + " W"
        summary += "\n"
        summary += "20 min: " + \
            str(np.around(np.max(self.moving_average(
                p_t, 1200 // self.road_profil["time_precision"])))) + " W"
        summary += "\n"
        summary += "Mean: " + str(np.around(np.mean(p_t))) + " W"
        summary += "\n"
        summary += "Normalized: " + \
            str(np.around(self.np_power(p_t))) + " W"

        self.split_summary = pd.DataFrame({"split start (km)": distance[:-1] * 1e-3,
                                           "split stop (km)": distance[1:] * 1e-3,
                                           "power (W)": p,
                                           "split (s)": self.compute_split_sum(time_split,
                                                                               idx),
                                           "speed (km/h)": self.compute_split_average(self.data_sampled["speed"].values,
                                                                                      idx) * 3.6,
                                           "gradient (%)": self.data_split["slope"].values,
                                           "wind (km/h)": 3.6 * self.data_split["wind"].values})

        bar = go.Bar(x=distance[:-1] + np.diff(x[idx]) // 2,
                     y=np.intp(p),
                     width=np.diff(x[idx]),
                     name="Power (W)",
                     hoverinfo="y+text+name",
                     hovertext=["Start {} km - End {} km - Duration {} s".format(np.around(j * 1e-3,
                                                                                           2),
                                                                                 np.around(distance[i + 1] * 1e-3,
                                                                                           2),
                                                                                 np.around(self.compute_split_sum(time_split,
                                                                                                                  idx)[i],
                                                                                           2)) for i,
                                j in enumerate(distance[:-1])])
        vel = go.Line(
            x=x,
            y=self.data_sampled["speed"].values * 3.6,
            name="Speed (km/h)",
            hoverinfo="y+text+name")
        ele = go.Line(
            x=x,
            y=elevation_sampled,
            name="Elevation (m)",
            hoverinfo="y+text+name")
        self.fig = go.Figure(data=[bar, vel, ele])
        self.fig.update_layout(
            template="seaborn",
            xaxis_title="Distance (m)",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x")
        self.fig.add_annotation(x=1.16, y=0.5, xref="paper", yref="paper",
                                text=summary.replace("\n",
                                                     "<br>"),
                                showarrow=False,
                                bordercolor="#c7c7c7",
                                bgcolor="#ffe680")
        route = go.Scattermapbox(
            lat=self.data_sampled["latitude"].values,
            lon=self.data_sampled["longitude"].values,
            mode='lines',
            marker=go.scattermapbox.Marker(size=10))

        position = go.Scattermapbox(
            lat=[self.data_sampled["latitude"].values[0]],
            lon=[self.data_sampled["longitude"].values[0]],
            mode='markers + text',
            marker=go.scattermapbox.Marker(size=10),
            textposition="bottom right",
            text=["Start"],
        )

        self.map_fig = go.Figure(data=[route, position])

        self.map_fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=self.data_sampled["latitude"].values[0],
                    lon=self.data_sampled["longitude"].values[0]),
                zoom=11),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            uirevision='constant'
        )

        self.app.layout = html.Div([
            dcc.Graph(
                id='map-plot',
                figure=self.map_fig,
                style={
                    'height': '30vh',
                    'width': '940px'}),
            dcc.Graph(
                id='main-plot',
                figure=self.fig,
                style={
                    'height': '50vh',
                    'width': '940px'}),
        ])

        return self.split_summary.to_json(orient="columns")
