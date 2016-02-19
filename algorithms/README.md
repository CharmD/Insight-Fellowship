# Insight_2016_Project

Therapy Prospector: Predicting Rehabilitation Effects for Constant Therapy Users (https://constanttherapy.com/)

The algorithms folder contains the code used to:
- Query the Constant Therapy SQL database 
- Visualize the data per patient cohort 
- Smoothen the behavioral data, namely the response accuracy and latency curves
- Select features of interest from user's demographic and baseline performance
- Predict user's end improvement on tasks through linear, radial SVM and random forest regression models in real time

Additional analyses classifies patient cohorts based on their behavioral data (accuracy and latency values, their improvement and variance) and patient characteristics (age, gender, deficits and time since diagnosis) in order to gain insight into differential task effects on patient groups. Pricipal components analysis and linear discriminant analysis were used.

The main code was written in Python: CT_FinalScript.py.
Example of regression models in R can be found: MachineLearning_Final.R