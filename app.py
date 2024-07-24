from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
import os
import gpxpy
from data import PeakPacer
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

app = Flask(__name__)
UPLOAD_FOLDER = './static/gpx'
app.config['ALLOWED_EXTENSIONS'] = {'gpx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "azerty"

dashapp = Dash(
    __name__,
    server=app,
    url_base_pathname='/dash-app/')

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
    tab = peak_pacer.analysis()

    return jsonify({'tab': tab, 'dash_url': '/dash-app'})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
