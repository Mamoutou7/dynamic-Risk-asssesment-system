"""
Name: app.py
Summary:
Module for setting up an API using

Author: Mamoutou Fofana
Date: 23/02/2023
"""
from flask import Flask, session, jsonify, request
import pandas as pd

import json
import os

import diagnostics
import scoring
######################Set up variables for use in our script


app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config["test_data_path"])
ingested_dataset_path = os.path.join(dataset_csv_path, "finaldata.csv")
test_dataset_path = os.path.join(test_data_path, "testdata.csv")

prediction_model = None

####################### Welcome Endpoint
@app.route('/')
def welcome():
    return "You're welcome to our model API !"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'GET', 'OPTIONS'])
def predict():        
    """
    call the prediction function you created in Step 3
    :return: The result of prediction
    """
    if request.method == 'POST':
        file = request.files['filename']
        dataset = pd.read_csv(file)
        # prediction performing
        predicted = diagnostics.model_predictions(dataset)
        # return value for prediction outputs
        return predicted

    if request.method == 'GET':
        file = request.args.get('filename')
        dataset = pd.read_csv(file)
        # prediction performing
        predicted = diagnostics.model_predictions(dataset)
        # return value for prediction outputs
        return {'predictions': str(predicted)}

    #######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """
    check the score of the deployed model
    :return: model score (f1 score)
    """
    return {'f1 score': scoring.score_model(test_dataset_path)}

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    """
    check means, medians, and modes for each column
    :return: list of all calculated summary statistics
    """
    return {'statistics': diagnostics.dataframe_summary()}

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostic():
    """
    check timing and percent NA values
    :return: value for all diagnostics
    """

    Pipeline = ["ingestion.py", "training.py", "diagnostics.py", "deployment.py"]
    execution_time = diagnostics.execution_time(Pipeline)
    missing_values = diagnostics.missing_data()
    outdated_packages_list = diagnostics.outdated_packages_list()

    return {
        "Execution time in seconds": execution_time,
        "Missing values in data": {col:nb for col, nb in
                                   zip(["lastmonth_activity", "lastyear_activity", "number_of_employees"],
                                       missing_values)},
        "outdated packages list": [{"Module": row[0],
                              "Version": row[1][0],
                              "Latest": row[1][1]}
                             for row in outdated_packages_list.iterrows()]
    }


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
