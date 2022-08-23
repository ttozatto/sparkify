# Sparkify - Churn Prediction for music streaming app with PySpark

This repository is part of the final project submited to Udacity for the Data Science Nanodegree.
The objective is to predict churn, from a simulated music streaming app, using historical data from user interactions.

A blog post with a detailed analysis is available at https://medium.com/@ttozatto.ds/churn-prediction-for-music-streaming-app-sparkify-d6e26d1ac80f

## Dependencies
  - pyspark
  - matplotlib
  
 ## Files
  - utils.py -> function to load and treat data, create, train and evaluate ML models
  - main.py -> script to run the full process, from loading the dataset to showing results
  - medium-sparkify-event-data.json -> dataset with user interactions in the app. Available at: https://video.udacity-data.com/topher/2018/December/5c1d6681_medium-sparkify-event-data/medium-sparkify-event-data.json
  
 ## Summary of Results
 ### Test Scores
 ![results_medium](https://user-images.githubusercontent.com/42552721/186053626-a014429d-c66c-485e-a418-b13b04d0345f.PNG)
 ### Parameters for best models
 ![bestModel](https://user-images.githubusercontent.com/42552721/186053668-d368dba2-c46e-419d-895e-f1e9ca88d1b5.PNG)
 ### Feature importance
![feature_importance](https://user-images.githubusercontent.com/42552721/186053678-ec77f392-a8b0-4134-9fbb-fa36dd1b19ae.png)

 
 ## Aknowledgements:
I would like to pay my special regards to:
  - Udacity, that proposed this work in the Data Science Nanodegree.
  - Spark team and community, that provides a powerful opensource tool to everyone.
