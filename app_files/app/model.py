# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:07:59 2016

@author: Charm
"""

# load packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.externals import joblib
from matplotlib import rcParams

# program constants
DEFAULT = "1"
EMPTY_STRING = ""
DEFAULT_LIST = [ 0.0 ]

TASK_LEVEL_1 = DEFAULT
TASK_LEVEL_2 = "2"
TASK_LEVEL_3 = "3"
TASK_LEVEL_4 = "4"
TASK_LEVEL_5 = "5"
REGRESSION_MODEL_LEVEL1 = "store/regressionModelLevel1.pkl"
REGRESSION_MODEL_LEVEL2 = "store/regressionModelLevel2.pkl"
REGRESSION_MODEL_LEVEL3 = "store/regressionModelLevel3.pkl"
REGRESSION_MODEL_LEVEL4 = "store/regressionModelLevel4.pkl"
REGRESSION_MODEL_LEVEL5 = "store/regressionModelLevel5.pkl"

GROUP_END_ACCURACY_LEVEL1 = "store/groupEndAccuracyLevel1.p"
GROUP_END_ACCURACY_LEVEL2 = "store/groupEndAccuracyLevel2.p"
GROUP_END_ACCURACY_LEVEL3 = "store/groupEndAccuracyLevel3.p"
GROUP_END_ACCURACY_LEVEL4 = "store/groupEndAccuracyLevel4.p"
GROUP_END_ACCURACY_LEVEL5 = "store/groupEndAccuracyLevel5.p"

PLOTTING_CONTEXT = "notebook"
PLOT_COLOR = "light blue"
LINE_COLOR = "r"
X_LABEL = "End Accuracy"

"""
    The Model for the accuracy improvement prediction.
"""
def Model(baselineAccuracy, gender, age, disorder, defecit, conditionSince, task = DEFAULT):

  # converting to numeric values
  age = int(age)
  baselineAccuracy = float(baselineAccuracy)
  
  # converting to arrays       
  gender = [int(c) for c in list(gender)]    
  disorder = [int(c) for c in list(disorder)]    
  defecit = [int(c) for c in list(defecit)]  
  conditionSince = [int(c) for c in list(conditionSince)]  
  
  # prepare input for the model
  modelInput = list()
  modelInput.extend(conditionSince)
  modelInput.extend(gender)
  modelInput.append(age)
  modelInput.extend(disorder)
  modelInput.extend(defecit)
  modelInput.append(baselineAccuracy)    
  arrayInput = np.array(modelInput, ndmin = 2)
  
  # load regression model
  model = None
  if task == TASK_LEVEL_1:
    model = REGRESSION_MODEL_LEVEL1
  elif task == TASK_LEVEL_2:
    model = REGRESSION_MODEL_LEVEL2
  elif task == TASK_LEVEL_3:
    model = REGRESSION_MODEL_LEVEL3
  elif task == TASK_LEVEL_4:
    model = REGRESSION_MODEL_LEVEL4
  elif task == TASK_LEVEL_5:
    model = REGRESSION_MODEL_LEVEL5  
  else:
    print("Unknow task level value entered: %s" % task)
    return DEFAULT_LIST
  
  regr = joblib.load(model)    
  return regr.predict(arrayInput)
  
"""
    The Grahp for the end accuracy prediction.
"""
def Graph(predictedEndAccuracy, task = DEFAULT):
  
    # load graph model
    graph = None
    if task == TASK_LEVEL_1:
        graph = GROUP_END_ACCURACY_LEVEL1
    elif task == TASK_LEVEL_2:
        graph = GROUP_END_ACCURACY_LEVEL2
    elif task == TASK_LEVEL_3:
        graph = GROUP_END_ACCURACY_LEVEL3
    elif task == TASK_LEVEL_4:
        graph = GROUP_END_ACCURACY_LEVEL4
    elif task == TASK_LEVEL_5:
        graph = GROUP_END_ACCURACY_LEVEL5
    else:
        print("Unknow task level value entered: %s" % task)
        return EMPTY_STRING
    
    groupEndAccuracy = pickle.load(open(graph, "rb"))
    
    # generate plot
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize = (5, 3.4), dpi = 100)
    ax = fig.add_subplot()    
    with sns.plotting_context(PLOTTING_CONTEXT, font_scale = 1.2):
        ax = sns.kdeplot(groupEndAccuracy, shade = True, color = sns.xkcd_rgb[PLOT_COLOR],  linewidth = 3);
         
    ax.set(yticks = [])
    ax.legend_.remove()
    ax.set_xlabel(X_LABEL, fontsize = 12)
    ax.hold(True)    
    plt.axvline(x = predictedEndAccuracy, ymin = 0, ymax = 1.2, linewidth = 2, color = LINE_COLOR)
    ax.set_xlim([0, 1.2])
    # calculate statistics
    sortedValues = pd.Series.sort_values(groupEndAccuracy)
    percentHigherThanPopulation = float(np.array(np.where(sortedValues < predictedEndAccuracy)).shape[1]) / len(sortedValues) * 100  
    
    return (fig, percentHigherThanPopulation)
    