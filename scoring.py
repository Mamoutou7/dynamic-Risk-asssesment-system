from flask import Flask, session, jsonify, request
import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from datetime import datetime


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_saved_path = os.path.join(config['output_model_path'])

# get current date
today = datetime.today().strftime('%Y%m%d')

#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model_path = os.path.join(model_saved_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    test_data = os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(test_data)

    X = test_data[["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1, 3)
    y = test_data['exited'].values.reshape(-1, 1)

    # evaluate model on test set
    predicted = model.predict(X)
    f1_score = metrics.f1_score(predicted, y)

    # save last score to disk
    score_file_path = os.path.join(model_saved_path, 'latestscore.txt')
    with open(score_file_path, 'w') as f:
        f.write(str(f1_score)) # f'{today}' + ' : '+ str(f1_score)

    return f1_score


if __name__ == '__main__':
    score_model()
