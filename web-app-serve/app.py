"""
Flask Serving
This file is a sample flask app that can be used to test your model with an API.
This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output
Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""
import os
from flask import Flask, send_file, request, render_template, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import zipfile
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import DataPreprocessing.preprocessing as preprocessing
import Detector.process_unet_output as detector
import shutil
from scipy.misc import bytescale
import Classifier.classifier as classifier

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'npy', 'zip'])
app = Flask(__name__)

@app.route('/<path:path>/')
def index(path):
    return render_template('index.html')


@app.route('/<path:path>/upload/', methods=["POST"])
def cancer_detection(path):
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        input_filepath = os.path.join('/output/', filename)
        file.save(input_filepath)
        zip_ref = zipfile.ZipFile(input_filepath, 'r')
        zip_ref.extractall('/output/DataPreprocessing_input')
        zip_ref.close()
        os.remove(input_filepath)
        os.makedirs('/output/DataPreprocessing_output/')
        os.makedirs('/output/Detector_output/')

        #preprocessing.full_prep('/output/DataPreprocessing_input/', '/output/DataPreprocessing_output/')
        #classifier.classifier('/output/Detector_output/')
        return render_template('preprocess.html')

@app.route('/<path:path>/preprocessPatient/')
def preprocess_patient(path):
    if len(os.listdir('/output/DataPreprocessing_input/')) == 0:
        return "No scans to preprocess!"
    if not os.path.exists('/output/Preprocessed_images/'):
        os.makedirs('/output/Preprocessed_images/')
    preprocessing.full_prep('/output/DataPreprocessing_input/', '/output/DataPreprocessing_output/')
    filename = os.listdir('/output/DataPreprocessing_output/')[0]
    patient = np.load(os.path.join('/output/DataPreprocessing_output/', filename))[0]
    scan_i = patient[patient.shape[0] // 2 + 1]
    scan_j = patient[patient.shape[0] // 2 + 10]
    img1 = Image.fromarray(scan_i)
    img2 = Image.fromarray(scan_j)
    img1.save('/output/Preprocessed_images/' + filename.replace('_clean.npy', '1.jpg'))
    img2.save('/output/Preprocessed_images/' + filename.replace('_clean.npy', '2.jpg'))
    filenames = [filename.replace('_clean.npy', '1.jpg'), filename.replace('_clean.npy', '2.jpg')]
    return render_template('detect.html', filenames = filenames)

@app.route('/<path:path>/<name>/getPreprocessedImage')
def get_preprocessed_image(path, name):
    filename = '/output/Preprocessed_images/' + name
    return send_file(filename, mimetype = 'image/jpg')

@app.route('/<path:path>/detectNodules/')
def nodules_detection(path):
    if len(os.listdir('/output/Detector_output/')) == 0:
        detector.detect_nodules('/output/DataPreprocessing_output/', '/output/Detector_output/')
    npy_files = os.listdir('/output/Detector_output/')
    if not os.path.exists('/output/Detector_images/'):
        os.makedirs('/output/Detector_images/')

    if len(npy_files) > 0:
        filenames = []
        for file in npy_files:
            arr = np.load(os.path.join('/output/Detector_output/', file))
            for i, scan in enumerate(arr):
                scan = bytescale(scan)
                img = Image.fromarray(scan)
                filename = file.replace('.npy',str(i)) + '.jpg'
                filenames.append(filename)
                img.save('/output/Detector_images/'+ filename)
            #json_arr = jsonify(filenames = filenames)
            return render_template('nodules.html', filenames = filenames)
    else:
        return "No scans to detect Nodules.."

@app.route('/<path:path>/cancerPredict/')
def cancer_prediction(path):
    prediction = classifier.classifier('/output/Detector_output/')
    prediction = prediction.tolist()[0]
    print(prediction)
    #return prediction
    #return render_template("prediction.html", prediction = prediction)
    
    return render_template("prediction.html", no_cancer_prob = prediction[0], cancer_prob = prediction[1])


@app.route('/<path:path>/<name>/getImage')
def return_image(path, name):
    filename = '/output/Detector_images/' + name
    return send_file(filename, mimetype = 'image/jpg')

@app.route('/<path:path>/clean/')
def clean_output(path):
    for folder in os.listdir('/output'):
        shutil.rmtree(os.path.join('/output', folder))
    return "Done"


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True, threaded = False)