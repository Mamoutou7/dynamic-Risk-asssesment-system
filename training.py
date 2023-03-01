import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import logging


# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/training.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

###################Load config.json and get path variables
logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])


def split_dataset(dataset):
    """
    segregate dataset into train set X and test set y
    :param dataset: dataset to into X and y
    :return: X (features to predict) and y (target)
    """
    # eliminate not used feature (corporation)
    X = dataset.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]].values.reshape(-1, 3)
    # target feature
    y = dataset['exited'].values.reshape(-1, 1).ravel()

    return X, y
#################Function for training the model
def train_model():
    """
    Train a logitic regression model for a dynamic risk assessment
    input: None
    :return: saved the trained model to disk
    """
    logging.info("Reading the training dataset")
    trainingdata_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    trainingdata = pd.read_csv(trainingdata_path)

    # split dataset into X and y
    X, y = split_dataset(trainingdata)

    # use this logistic regression for training
    logging.info("Using this logistic regression for training")
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # fit the logistic regression to your data
    model = logit.fit(X, y)

    logging.info("Saving the model in saving_model_path")
    # path to save model on the disk
    saving_model_path = os.path.join(model_path, 'trainedmodel.pkl')
    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(saving_model_path, 'wb') as f:
        pickle.dump(model, f)



if __name__ == '__main__':
    train_model()