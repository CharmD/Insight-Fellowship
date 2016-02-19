# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:07:59 2016

@author: Charm
"""
#load packages
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#change directory to where my app files ares
import os
os.chdir("/Users/Charm/OneDrive/app_files/store")

GroupEndAccuracy = pickle.load( open( "groupEndAccuracyLevel5.p", "rb" ) )


fig = Figure(figsize=(5,4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvas(fig)
with sns.plotting_context("notebook",font_scale=1.2):
     #ax = sns.distplot(GroupEndAccuracy, hist=False, rug=True, color='blue');
     ax = sns.kdeplot(GroupEndAccuracy, shade=True, color = sns.xkcd_rgb["light blue"],  linewidth=3);
ax.set(yticks=[])
ax.legend_.remove()
ax.set_xlabel('End Accuracy', fontsize=16)
ax.hold(True)
ax.set_title('Level5', fontsize=16)

predictedEnd = 0.85 #value output by my model from the App
plt.axvline(x=predictedEnd, ymin=0, ymax = 1.2, linewidth=2, color='r')
aa = pd.Series.sort_values(GroupEndAccuracy)
HigherThanPopulation = float(np.array(np.where(aa<predictedEnd)).shape[1])/len(aa)*100

print("Patient is likely to perform better than %d percent of the population on this task" %HigherThanPopulation)


