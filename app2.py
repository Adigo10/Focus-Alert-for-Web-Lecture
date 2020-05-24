# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:13:55 2020

@author: Vinayak
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


       


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict',methods=['GET'])
def opencv():
    model()    ##CAllING  FUNCTION FROM FOCUS.PY


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
