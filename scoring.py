import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from datetime import datetime
import logging

from training import split_dataset

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/scoring.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#################Load config.json and get path variables
logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_saved_path = os.path.join(config['output_model_path'])

# get current date
today = datetime.today().strftime('%Y%m%d')

#################Function for model scoring
def score_model(test_data):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    logging.info("Defining the model saved path variable")
    model_path = os.path.join(model_saved_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)


    test_data = pd.read_csv(test_data)

    # split testdata into X and y
    X, y = split_dataset(test_data)

    # evaluate model on test set
    predicted = model.predict(X)
    f1_score = metrics.f1_score(predicted, y)

    # save last score to disk
    logging.info("Defining the last score path variable")
    score_file_path = os.path.join(model_saved_path, 'latestscore.txt')

    logging.info("Writing the model score to latestscore.txt")
    with open(score_file_path, 'w') as f:
        f.write(str(f1_score))

    return f1_score


if __name__ == '__main__':
    test_data = os.path.join(test_data_path, 'testdata.csv')
    score_model(test_data)
