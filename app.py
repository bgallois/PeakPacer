from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename
import os
import gpxpy
from data import PeakPacer
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = './static/gpx'
app.config['ALLOWED_EXTENSIONS'] = {'gpx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = str(uuid.uuid4())

user_data = {}

dashapp = Dash(
    __name__,
    server=app,
    url_base_pathname='/dash-app/')


def get_session_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


dashapp.layout = html.Div([
    dcc.Graph(
        id='map-plot',
        figure=go.Figure(),
        style={
            'height': '30vh',
            'width': '940px'}),
    dcc.Graph(
        id='main-plot',
        figure=go.Figure(),
        style={
            'height': '50vh',
            'width': '940px'}),
    dcc.Store(id="session-id"),
])


@dashapp.callback(
    Output(
        'map-plot',
        'figure',
        allow_duplicate=True),
    Input('main-plot', 'hoverData'),
    [State('session-id', 'data'), State('map-plot', 'figure')],
    prevent_initial_call=True,
)
def update_map(hoverData, session_id, map_fig):
    if session_id not in user_data:
        return map_fig

    print(session_id)
    data = user_data[session_id]["data_sampled"]

    if hoverData is None or data is None:
        return map_fig

    updated_map_fig = go.Figure(map_fig)
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


peak_pacer = PeakPacer(dashapp)


def allowed_file(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/get-coordinates')
def get_coordinates():
    data = {
        "latitude": 37.7749,
        "longitude": -122.4194
    }
    return jsonify(data)


@app.route('/')
def index():
    return render_template('map.html')


def process_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        coordinates = []
        elevations = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    coordinates.append([point.latitude, point.longitude])
                    elevations.append(point.elevation)
    return coordinates, elevations


@app.route('/submit-data', methods=['POST'])
def process_data():
    gpx = request.files.get('fileInput')
    ftp = request.form.get('ftp')
    pma_value = request.form.get('pma')
    total_weight = request.form.get('totalWeight')
    efficiency = request.form.get('efficiency')
    cda = request.form.get('cda')
    rolling_friction = request.form.get('rollingFriction')
    air_density = request.form.get('airDensity')
    wind_direction = request.form.get('windDirection')
    wind_speed = request.form.get('windSpeed')

    if not all([ftp, pma_value, total_weight, efficiency, cda,
               cda, rolling_friction, air_density, wind_direction, wind_speed]):
        return jsonify({'error': 'All fields are required'}), 400

    if gpx and allowed_file(gpx.filename):
        filename = secure_filename(gpx.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        gpx.save(filepath)
    else:
        gpx.filename = os.path.join("demo.gpx")

    rider_profil = {
        'ftp': float(ftp),
        'pma': float(pma_value),
        'masse': float(total_weight),
        'efficiency': float(efficiency) / 100,
        'cda': float(cda)
    }

    road_profil = {
        "air_density": float(air_density),
        "rolling_friction": float(rolling_friction),
        "wind_direction_param": float(wind_direction),
        "wind_speed_param": float(wind_speed) / 3.6,
        "coordinates": process_gpx(os.path.join(app.config['UPLOAD_FOLDER'], gpx.filename))[0],
        "elevation": process_gpx(os.path.join(app.config['UPLOAD_FOLDER'], gpx.filename))[1],
    }
    peak_pacer.set_parameters(rider_profil, road_profil)
    data_split, data_sampled, tab, summary = peak_pacer.analysis()
    user_data[get_session_id()] = {"data_split": data_split,
                                   "data_sampled": data_sampled,
                                   "tab": tab,
                                   "summary": summary}
    plot(data_split, data_sampled, summary)

    return jsonify({'tab': tab})


def plot(data_split, data_sampled, summary):
    distance = np.insert(data_split["dx"].values, 0, 0)
    distance = np.cumsum(distance)
    bar = go.Bar(x=distance[:-1] + np.diff(distance) // 2,
                 y=np.intp(data_split["power"].values),
                 width=np.diff(distance),
                 name="Power (W)",
                 hoverinfo="y+text+name",
                 hovertext=["Start {} km - End {} km - Duration {} s".format(np.around(j * 1e-3,
                                                                                       2),
                                                                             np.around(distance[i + 1] * 1e-3,
                                                                                       2),
                                                                             np.around(data_split["time"].values[i],
                                                                                       2)) for i,
                            j in enumerate(distance[:-1])])
    vel = go.Line(
        x=data_sampled["distance"].values,
        y=data_sampled["speed"].values * 3.6,
        name="Speed (km/h)",
        hoverinfo="y+text+name")
    ele = go.Line(
        x=data_sampled["distance"].values,
        y=data_sampled["elevation"].values,
        name="Elevation (m)",
        hoverinfo="y+text+name")
    fig = go.Figure(data=[bar, vel, ele])
    fig.update_layout(
        template="seaborn",
        xaxis_title="Distance (m)",
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="x")
    fig.add_annotation(x=1, y=0, xref="paper", yref="paper", xanchor='left', yanchor='bottom',
                       text=summary.replace("\n",
                                            "<br>"),
                       showarrow=False,
                       bordercolor="#c7c7c7",
                       bgcolor="#ffe680")
    route = go.Scattermapbox(
        lat=data_sampled["latitude"].values,
        lon=data_sampled["longitude"].values,
        mode='lines',
        marker=go.scattermapbox.Marker(size=10))

    position = go.Scattermapbox(
        lat=[data_sampled["latitude"].values[0]],
        lon=[data_sampled["longitude"].values[0]],
        mode='markers + text',
        marker=go.scattermapbox.Marker(size=10),
        textposition="bottom right",
        text=["Start"],
    )

    map_fig = go.Figure(data=[route, position])

    map_fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=data_sampled["latitude"].values[0],
                lon=data_sampled["longitude"].values[0]),
            zoom=11),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision='constant'
    )

    dashapp.layout = html.Div([
        dcc.Graph(
            id='map-plot',
            figure=map_fig,
            style={
                'height': '35vh',
                'margin': '0',
            }),
        dcc.Graph(
            id='main-plot',
            figure=fig,
            style={
                'height': '55vh',
                'margin': '0',
            }),
        dcc.Store(id="session-id", data=get_session_id()),
    ])


@dashapp.server.before_request
def restrict_access():
    if request.path == '/dash-app/':
        referer = request.headers.get('Referer')
        if not referer:
            return "Access Denied", 403


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
