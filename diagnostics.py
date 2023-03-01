"""
Name: diagnostics.py
Summary:
Module for diagnosing the Machine learning pipeline
Author: Mamoutou Fofana
Date: 23/02/2023
"""
import pickle
import pandas as pd
import numpy as np
import time
import os
import json
import subprocess
from io import StringIO

from training import split_dataset


##################Load config.json and get environment variables

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset=None):
    # read the deployed model and a test dataset, calculate predictions

    if dataset is None:
        dataset_path = os.path.join(test_data_path, 'testdata.csv')
        dataset = pd.read_csv(dataset_path)

    # load the deployed model
    prod_deployment_path = os.path.join(model_deployment_path, 'trainedmodel.pkl')
    with open(prod_deployment_path, 'rb') as f:
        model = pickle.load(f)

    # split testdata into X and y
    X, y = split_dataset(dataset)

    # evaluate model on test set
    preds = model.predict(X)

    # return value should be a list containing all predictions
    return preds

##################Function to get summary statistics
def dataframe_summary():
    """
    calculate summary statistics here
    :param: None
    :return: value should be a list containing all summary statistics
    """
    # construct datatset path
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    # read dataset
    dataset = pd.read_csv(dataset_path)

    statistics = []
    # numeric columns
    numeric_column_idx = np.where(dataset.dtypes != object)[0]
    numeric_column = dataset.columns[numeric_column_idx].tolist()
    means = dataset[numeric_column].mean().to_list()
    stds = dataset[numeric_column].std().to_list()
    medians = dataset[numeric_column].quantile().to_list()

    statistics.append(means)
    statistics.append(medians)
    statistics.append(stds)

    return statistics

##################Function to get missing data
def missing_data():
    """calculate missing data on the dataset
    return % of missing data per column
    """
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    dataset = pd.read_csv(dataset_path)

    missing_data = dataset.isna().sum(axis=0)
    missing_data /= len(dataset)*100

    return missing_data.to_list()

##################Function to get timings
def execution_time(filenames):
    """
    calculate timing of training.py and ingestion.py
    :param filenames: list of files names in filenames to run and to calculate their execution durations
    :return: [dictionary] A dictionary with execution duration of file in filenames
    """
    execution_durations = {}
    for file in filenames:

        start_time = time.time()
        os.system("python "+str(file))
        end_time = time.time() - start_time
        execution_durations[file] = end_time
        # return a list of 2 timing values in seconds

    return execution_durations

def cmd_output_df(cmd):
    """

    :param cmd: command to execute (pip list)
    :return: output of command in pandas dataframe
    """
    a = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    b = StringIO(a.communicate()[0].decode('utf-8'))
    df = pd.read_csv(b, sep="\s+")

    df.drop(index=[0], axis=0, inplace=True)
    df = df.set_index('Package')

    return df

##################Function to check dependencies
def outdated_packages_list():

    # get all dependencies
    installed_cmd = ['pip', 'list']
    current_dependencies = cmd_output_df(installed_cmd)
    current_dependencies = current_dependencies.rename(columns={'Version': 'Latest'})


    # get all dependencies outdated
    broken_cmd = ['pip', 'list', '--outdated']
    outdated_dependencies = cmd_output_df(broken_cmd)
    outdated_dependencies.drop(['Version', 'Type'], axis=1, inplace=True)

    # dataframe containing pip list
    requirements = pd.read_csv('requirements.txt', sep='==', header=None, names=['Package', 'Version'], engine='python')
    requirements = requirements.set_index('Package')

    dependencies = requirements.join(current_dependencies)
    for p in outdated_dependencies.index:
        if p in dependencies.index:
            dependencies.loc[p, 'Latest'] = outdated_dependencies.loc[p, 'Latest']

    # keep only outdated dependencies (ie latest version exists)
    dependencies.dropna(inplace=True)

    return dependencies


if __name__ == '__main__':
    model_predictions(dataset=None)
    dataframe_summary()
    missing_data()
    filenames = ["training.py", "ingestion.py"]
    execution_time(filenames)
    outdated_packages_list()





    
