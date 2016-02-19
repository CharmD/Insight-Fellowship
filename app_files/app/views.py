# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:43:43 2016

@author: Charm
"""

# load packages
from flask import render_template, request, make_response, jsonify
from app import app
from model import Model, Graph
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import StringIO
import base64

# program constants
TASK = "task"
DEFAULT_TASK = "1"

BASELINE_ACCURACY = "baselineAccuracy"
DEFAULT_BASELINE_ACCURACY = "0.5"

GENDER = "gender"
DEFAULT_GENDER = "10"

AGE = "age"
DEFAULT_AGE = "55"

DISORDER = "disorder"
DEFAULT_DISORDER = "100000"

DEFICIT = "defecit"
DEFAULT_DEFICIT = "1000"

CONDITION_SINCE = "conditionSince"
DEFAULT_CONDITION_SINCE = "100000"

FLOAT_FORMAT = "%.2f"
INDEX_PAGE = "index.html"

CONTENT_TYPE_ATTRIBUTE = "Content-Type"
CONTENT_TYPE_VALUE = "image/png"

PREDICTION_STRING_FORMAT_LINE1 = "Predicted end accuracy for this patient is %s."
PREDICTION_STRING_FORMAT_LINE2 = "Patient is likely to perform better than %s of the population on this task."

MESSAGE_KEY_LINE1 = "messageLine1"
MESSAGE_KEY_LINE2 = "messageLine2"
GRAPH_KEY = "graph"
PERCENT = "%"

"""
    Computes the end accuracy based on the input features provided.
"""
def computeEndAccuracy(task):
        
    # pull 'BASELINE_ACCURACY' from input field and store it
    baselineAccuracy = request.args.get(BASELINE_ACCURACY)
    if not baselineAccuracy:
        baselineAccuracy = DEFAULT_BASELINE_ACCURACY
        
    # pull 'GENDER' from input field and store it
    gender = request.args.get(GENDER)
    if not gender:
        gender = DEFAULT_GENDER
    
    # pull 'AGE' from input field and store it
    age = request.args.get(AGE)
    if not age:
        age = DEFAULT_AGE
        
    # pull 'DISORDER' from input field and store it
    disorder = request.args.get(DISORDER)
    if not disorder:
        disorder = DEFAULT_DISORDER
    
    # pull 'DEFICIT' from input field and store it
    defecit = request.args.get(DEFICIT)
    if not defecit:
        defecit = DEFAULT_DEFICIT
        
    # pull 'CONDITION_SINCE' from input field and store it
    conditionSince = request.args.get(CONDITION_SINCE)
    if not conditionSince:
        conditionSince = DEFAULT_CONDITION_SINCE
    
    # generate prediction based on input    
    predictedEndAccuracy = Model(baselineAccuracy, gender, age, disorder, defecit, conditionSince, task)
    
    return predictedEndAccuracy[0]
    
"""
    Generated graph in '.png' format based on the input features provided.
"""
def generateGraphImage(task):
    
    # generate prediction and graph based on input    
    predictedEndAccuracy = computeEndAccuracy(task)    
    graphData = Graph(predictedEndAccuracy, task)
        
    # create image from the figure and return it back to the client
    canvas = FigureCanvas(graphData[0])
    pngOutput = StringIO.StringIO()
    canvas.print_png(pngOutput)
    return pngOutput.getvalue()

@app.route('/')
@app.route('/index')
def index():
    
    # pull 'TASK' from input field and store it
    task = request.args.get(TASK)
    if not task:
        task = DEFAULT_TASK
    
    # generate prediction based on input    
    predictedEndAccuracy = computeEndAccuracy(task)    
    floatVal =  FLOAT_FORMAT % predictedEndAccuracy    
    
    graphData = Graph(predictedEndAccuracy, task)
    popPercentVal = FLOAT_FORMAT % graphData[1]       
    
    return render_template(INDEX_PAGE, endAccuracy = floatVal, popPercent = popPercentVal)
    
@app.route('/figure')
def figure():
    
    # pull 'TASK' from input field and store it
    task = request.args.get(TASK)
    if not task:
        task = DEFAULT_TASK
    
    # generate graph based on input        
    png = generateGraphImage(task)
    response = make_response(png)
    response.headers[CONTENT_TYPE_ATTRIBUTE] = CONTENT_TYPE_VALUE
    
    return response
    
@app.route('/result')
def result():
    
    # pull 'TASK' from input field and store it
    task = request.args.get(TASK)
    if not task:
        task = DEFAULT_TASK        
    
    # generate prediction and graph based on input    
    predictedEndAccuracy = computeEndAccuracy(task)
    floatVal =  (FLOAT_FORMAT % predictedEndAccuracy)
    
    graphData = Graph(predictedEndAccuracy, task)
    popPercentVal = (FLOAT_FORMAT % graphData[1]) + PERCENT
    
    messageLine1 =  PREDICTION_STRING_FORMAT_LINE1 % floatVal
    messageLine2 =  PREDICTION_STRING_FORMAT_LINE2 % popPercentVal
    
    png = generateGraphImage(task)
    encodedString = base64.b64encode(png)    
    
    # generate response object to return via JSON
    response = { MESSAGE_KEY_LINE1 : messageLine1, MESSAGE_KEY_LINE2 : messageLine2, GRAPH_KEY : encodedString }                
    return jsonify(result = response) 