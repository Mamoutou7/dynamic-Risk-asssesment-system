"""
Name: fullprocess.py
Summary:
Module for calling functions from scripts that you wrote in previous steps
Author: Mamoutou Fofana
Date: 28/02/2023
"""


import ast
import os
import json
import pandas as pd
from sklearn import metrics

import deployment
import reporting
import training
import diagnostics
from scoring import score_model

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = os.path.join(config['input_folder_path'])
output_model_path = os.path.join(config['output_model_path'])

ingested_file_path = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
latest_score_path = os.path.join(prod_deployment_path, 'latestscore.txt')



##################Check and read new data
# first, read ingestedfiles.txt

with open(ingested_file_path, 'r') as f:
    ingested_files = ast.literal_eval(f.read())

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    file_found_flag = False
    for file in os.listdir(input_folder_path):
        if file not in ingested_files:
            print(file)
            file_found_flag = True

    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if file_found_flag:
        os.system("python ingestion.py")
    else:
        print("No new data ingested, exiting....")
        exit()

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(latest_score_path, 'r') as file:
    latest_score = float(file.read())

final_data_path = os.path.join(dataset_csv_path,'finaldata.csv')
dataset = pd.read_csv(final_data_path)
new_preds = diagnostics.model_predictions(dataset)

y = dataset['exited']

new_score = metrics.f1_score(y, new_preds)

print(f'latest score: {latest_score}, new score: {new_score}')

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if new_score < latest_score:
     print("Proceed: Model drift found")
else:
    print("Proceed: No new data ingested then model drift not found")
    exit()


if file_found_flag:
     training.train_model()

# ##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
if file_found_flag:
    print('deploying new model')
    deployment.store_model_into_pickle()

# ##################Diagnostics and reporting
# # run diagnostics.py and reporting.py for the re-deployed model
if file_found_flag:
    print('producing reporting and calling apis for statistics')
    os.system("python diagnostics.py")
    reporting.score_model()
    os.system("python apicalls.py")
