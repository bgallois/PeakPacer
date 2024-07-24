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


def aero_force(velocity, air_density, cda, wind_speed):
    return 0.5 * cda * air_density * (velocity + wind_speed)**2


def rolling_resistance(rolling_friction, masse, slope):
    return rolling_friction * masse * 9.8067 * np.cos(np.arctan(slope * 0.01))


def gravity(masse, slope):
    return masse * 9.8067 * np.sin(np.arctan(slope * 0.01))


def propulsive_power(velocity, air_density, cda, wind_speed,
                     rolling_friction, masse, slope, efficiency, **kwargs):
    return velocity * efficiency * (aero_force(velocity, air_density, cda, wind_speed) +
                                    rolling_resistance(rolling_friction, masse, slope) + gravity(masse, slope))


def velocity_from_power(power, air_density, cda, wind_speed,
                        rolling_friction, masse, slope, efficiency, **kwargs):

    order_3 = np.ones(len(power)) * efficiency * 0.5 * air_density * cda
    order_2 = np.ones(len(power)) * efficiency * air_density * cda * wind_speed
    order_1 = efficiency * (0.5 * air_density * cda * wind_speed +
                            rolling_resistance(rolling_friction, masse, slope) + gravity(masse, slope))
    order_0 = -power

    coefficients = np.vstack([order_3, order_2, order_1, order_0]).T
    roots = [np.roots(coeff) for coeff in coefficients]
    velocities = np.array(
        [np.max(r[np.isreal(r) & (r >= 0)]).real for r in roots])
    return velocities


def moving_average(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def target_power(x, rider_profil):

    def func(x, a, b, c):
        return (rider_profil["ftp"]) + a / x

    fatigue = 1

    popt, pcov = scipy.optimize.curve_fit(func, [3600, 300, 60],
                                          [rider_profil["ftp"], rider_profil["pma"] * fatigue, 1.8 * rider_profil["pma"] * fatigue])
    return int(func(x, *popt))


def px2pt(power, road_profil, rider_profil):
    t = np.cumsum(
        road_profil["dist_precision"] /
        velocity_from_power(
            power=power,
            slope=road_profil["slope_sampled"],
            wind_speed=road_profil["wind_speed_sampled"],
            **road_profil,
            **rider_profil))
    x = np.arange(0, t[-1], road_profil["time_precision"])
    p_resamp = np.interp(x, t, power)
    return p_resamp


def np_power(power, road_profil, rider_profil):
    power = moving_average(power, 30 // road_profil["time_precision"])
    return np.mean(power**4)**(1 / 4)


def travel_time(power, dx, road_profil, rider_profil):
    t = np.sum(
        dx /
        velocity_from_power(
            power=power,
            slope=road_profil["slope_split"],
            wind_speed=road_profil["wind_speed_split"],
            **road_profil,
            **rider_profil))
    return t


def mean_power_constraint(power, dx, road_profil, rider_profil):
    t = dx / velocity_from_power(power=power,
                                 slope=road_profil["slope_split"],
                                 wind_speed=road_profil["wind_speed_split"],
                                 **road_profil,
                                 **rider_profil)
    return target_power(np.sum(t), rider_profil) - \
        np.sum(t * power) / np.sum(t)


def pma_power_constraint(power, dx, road_profil, rider_profil):
    t = dx / velocity_from_power(power=power,
                                 slope=road_profil["slope_split"],
                                 wind_speed=road_profil["wind_speed_split"],
                                 **road_profil,
                                 **rider_profil)

    p_time = []
    for i, j in enumerate(t):
        p_time.extend([power[i]] * int(j / 10))
    return 0.95 * rider_profil["pma"] - np.max(moving_average(p_time, 30))


def minimize_time(distance, rider_profil, road_profil, it=100):
    dx = np.diff(distance)
    start = np.random.normal(
        rider_profil["ftp"], 50, len(
            road_profil["slope_split"]))
    mean_power = ({'type': 'ineq', 'fun': mean_power_constraint,
                  'args': (dx, road_profil, rider_profil)})
    pma_power = ({'type': 'ineq', 'fun': pma_power_constraint,
                 'args': (dx, road_profil, rider_profil)})

    A = scipy.optimize.minimize(
        travel_time,
        start,
        args=(
            dx,
            road_profil,
            rider_profil),
        options={
            "eps": 2e-2,
            "maxiter": it,
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


def wind(orientation, wind_direction, wind_force):
    wind_direction = np.ones_like(orientation) * wind_direction
    return wind_force * np.cos(np.deg2rad(orientation - wind_direction))


def compute_split_average(values, splits):
    means = []
    for i, j in enumerate(splits[:-1]):
        means.append(np.mean(values[j:splits[i + 1]]))
    return np.asarray(means)


def compute_split_sum(values, splits):
    means = []
    for i, j in enumerate(splits[:-1]):
        means.append(np.sum(values[j:splits[i + 1]]))
    return np.asarray(means)


def analysis(rider_profil, road_profil, app):
    print("Start Analysis")
    road_profil["dist_precision"] = 10
    road_profil["time_precision"] = 10
    geod = Geodesic.WGS84
    coords = road_profil["coordinates"]
    road_profil["distance"] = np.cumsum(
        [0] + [geod.Inverse(*j, *coords[i + 1])['s12'] for i, j in enumerate(coords[:-1])])
    road_profil["direction"] = [
        0] + [geod.Inverse(*j, *coords[i + 1])['azi1'] for i, j in enumerate(coords[:-1])]

    x = np.arange(0, road_profil["distance"][-1],
                  road_profil["dist_precision"])

    elevation_sampled = np.interp(
        x, road_profil["distance"], road_profil["elevation"])
    direction_sampled = np.interp(
        x, road_profil["distance"], road_profil["direction"])
    slope_sampled = np.diff(elevation_sampled) / np.diff(x) * 100
    slope_sampled = np.insert(slope_sampled, 0, slope_sampled[0])

    latitude_sampled = np.interp(
        x, road_profil["distance"], [i for i, _ in coords])
    longitude_sampled = np.interp(
        x, road_profil["distance"], [i for _, i in coords])

    road_profil["slope_sampled"] = slope_sampled
    road_profil["wind_speed_sampled"] = wind(
        direction_sampled,
        np.ones_like(direction_sampled) *
        road_profil["wind_direction_param"],
        road_profil["wind_speed_param"])

    splits = np.sign(np.diff(scipy.signal.wiener(elevation_sampled, 5)))
    splits[splits == 0] = 1
    idx = np.where(splits[:-1] != splits[1:])[0]
    idx[0] = 0  # np.insert(idx, 0, 0)
    direction_change = np.abs((np.diff(direction_sampled) + 180) % 360 - 180)
    peaks, _ = scipy.signal.find_peaks(direction_change, height=40)
    idx = np.unique(np.sort(np.append(idx, peaks)))
    idx = idx[np.where(np.diff(idx) > 5)[0]]
    idx = np.append(idx, len(x) - 1)

    distance = x[idx]
    road_profil["slope_split"] = compute_split_average(
        road_profil["slope_sampled"], idx)
    road_profil["wind_speed_split"] = compute_split_average(
        road_profil["wind_speed_sampled"], idx)

    p, t = minimize_time(distance, rider_profil, road_profil, 1000)

    p_sampled = np.zeros_like(x)
    for i, j in enumerate(x):
        bound = np.argmax(j < distance)
        p_sampled[i] = p[bound - 1]

    velocity_sampled = velocity_from_power(
        power=p_sampled,
        slope=road_profil["slope_sampled"],
        wind_speed=road_profil["wind_speed_sampled"],
        **road_profil,
        **rider_profil)
    time_split = np.diff(x) / velocity_sampled[:-1]
    time_exact = np.sum(np.diff(x) / velocity_sampled[:-1])
    p_t = px2pt(p_sampled, road_profil, rider_profil)

    summary = "Time: " + str(datetime.timedelta(seconds=time_exact))
    summary += "\n"
    summary += "Speed: " + \
        str(np.around(distance[-1] / time_exact * 3.6)) + " km/h"
    summary += "\n"
    summary += "1 min: " + \
        str(np.around(np.max(moving_average(
            p_t, 60 // road_profil["time_precision"])))) + " W"
    summary += "\n"
    summary += "5 min: " + \
        str(np.around(np.max(moving_average(
            p_t, 300 // road_profil["time_precision"])))) + " W"
    summary += "\n"
    summary += "8 min: " + \
        str(np.around(np.max(moving_average(
            p_t, 480 // road_profil["time_precision"])))) + " W"
    summary += "\n"
    summary += "10 min: " + \
        str(np.around(np.max(moving_average(
            p_t, 600 // road_profil["time_precision"])))) + " W"
    summary += "\n"
    summary += "20 min: " + \
        str(np.around(np.max(moving_average(
            p_t, 1200 // road_profil["time_precision"])))) + " W"
    summary += "\n"
    summary += "Mean: " + str(np.around(np.mean(p_t))) + " W"
    summary += "\n"
    summary += "Normalized: " + \
        str(np.around(np_power(p_t, road_profil, rider_profil))) + " W"

    split_summary = pd.DataFrame({"split start (km)": distance[:-1] * 1e-3,
                                  "split stop (km)": distance[1:] * 1e-3,
                                  "power (W)": p,
                                  "split (s)": compute_split_sum(time_split,
                                                                 idx),
                                  "speed (km/h)": compute_split_average(velocity_sampled,
                                                                        idx) * 3.6,
                                  "gradient (%)": road_profil["slope_split"],
                                  "wind (km/h)": 3.6 * road_profil["wind_speed_split"]})

    bar = go.Bar(x=distance[:-1] + np.diff(x[idx]) // 2,
                 y=np.intp(p),
                 width=np.diff(x[idx]),
                 name="Power (W)",
                 hoverinfo="y+text+name",
                 hovertext=["Start {} km - End {} km - Duration {} s".format(np.around(j * 1e-3,
                                                                                       2),
                                                                             np.around(distance[i + 1] * 1e-3,
                                                                                       2),
                                                                             np.around(compute_split_sum(time_split,
                                                                                                         idx)[i],
                                                                                       2)) for i,
                            j in enumerate(distance[:-1])])
    vel = go.Line(
        x=x,
        y=velocity_sampled * 3.6,
        name="Speed (km/h)",
        hoverinfo="y+text+name")
    ele = go.Line(
        x=x,
        y=elevation_sampled,
        name="Elevation (m)",
        hoverinfo="y+text+name")
    fig = go.Figure(data=[bar, vel, ele])
    fig.update_layout(
        template="seaborn",
        xaxis_title="Distance (m)",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x")
    fig.add_annotation(x=1.16, y=0.5, xref="paper", yref="paper",
                       text=summary.replace("\n",
                                            "<br>"),
                       showarrow=False,
                       bordercolor="#c7c7c7",
                       bgcolor="#ffe680")
    route = go.Scattermapbox(
        lat=latitude_sampled,
        lon=longitude_sampled,
        mode='lines',
        marker=go.scattermapbox.Marker(size=10))

    position = go.Scattermapbox(
        lat=[latitude_sampled[0]],
        lon=[longitude_sampled[0]],
        mode='markers + text',
        marker=go.scattermapbox.Marker(size=10),
        textposition="bottom right",
        text=["Start"],
    )

    map_fig = go.Figure(data=[route, position])

    map_fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=latitude_sampled[0], lon=longitude_sampled[0]),
            zoom=11),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision='constant'
    )

    # Trick to not have same callback
    # By redefining the plot every time
    uid = uuid.uuid4()
    app.layout = html.Div([
        dcc.Graph(
            id='map-plot{}'.format(uid),
            figure=map_fig,
            style={
                'height': '30vh',
                'width': '100vw',
                'margin-right': 'auto',
                'margin-left': 'auto'}),
        dcc.Graph(
            id='main-plot{}'.format(uid),
            figure=fig,
            style={
                'height': '50vw',
                'width': '100vw'}),
    ])

    @app.callback(
        Output(
            'map-plot{}'.format(uid),
            'figure',
            allow_duplicate=True),
        # TODO move callback to app
        Input('main-plot{}'.format(uid), 'hoverData'),
        prevent_initial_call=True
    )
    def update_map(hoverData):
        if hoverData is None:
            return map_fig

        latitude = latitude_sampled
        longitude = longitude_sampled
        p = p_sampled
        velocity = velocity_sampled
        wind = road_profil["wind_speed_sampled"]
        slope = road_profil["slope_sampled"]

        updated_map_fig = map_fig
        index = hoverData["points"][0]["pointIndex"]
        updated_map_fig.data[1].lat = [latitude[index]]
        updated_map_fig.data[1].lon = [longitude[index]]

        updated_map_fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=latitude[index], lon=longitude[index]),
            ))
        text = "Power: {} W <br> Speed: {} km/h <br> Wind: {} km/h <br> Slope: {}".format(
            np.around(p[index]), np.around(velocity[index]), np.around(wind[index]), np.around(slope[index]))
        updated_map_fig.data[1].text = text
        return updated_map_fig

    print("Finished Analysis")
    return app, split_summary.to_json(orient="columns")
