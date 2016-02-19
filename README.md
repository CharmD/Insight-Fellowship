# Insight_2016_Project

This project was created for the Insight 2016 fellowship using data obtained from the Constant Therapy app: https://constanttherapy.com/

Therapy Prospector: Predicting Rehabilitation Effects for Constant Therapy Users 

The "algorithms" folder contains the code used to:
- Query the Constant Therapy SQL database 
- Visualize the data per patient cohort 
- Smoothen the behavioral data, namely the response accuracy and latency curves
- Select features of interest from user's demographic and baseline performance
- Predict user's end improvement on tasks through linear, radial SVM and random forest regression models in real time

Additional analyses classifies patient cohorts based on their behavioral data (accuracy and latency values, their improvement and variance) and patient characteristics (age, gender, deficits and time since diagnosis) in order to gain insight into differential task effects on patient groups. Pricipal components analysis and linear discriminant analysis were used.

The main code was written in Python: CT_FinalScript.py.
Example of regression models in R can be found: MachineLearning_Final.R


The "app_files" folder contains the code for the interactive front-­‐end using Flask, Bootstrap css, HTML, jQuery, JSON and AJAX: www.therapyprospector.me

therapyprospector.me predicts the effects of therapy for new constant therapy patients, offering them more effective, targeted and personalized brain rehabilitation. 
It predicts the patient's end performance on a set task and difficulty level based on their demographic information and preliminary baseline activity.
It compares their predicted end performance to the population of constant therapy patients.