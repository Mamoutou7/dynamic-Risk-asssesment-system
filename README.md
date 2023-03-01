## Dynamic Risk Assessment System

This project is part of Udacity Machine Learning DevOps engineer Nanodegree Program

Author: Mamoutou FOFANA

Date: 01/03/2023

## Project Overview: Training, Scoring, and Deploying an ML Model
With an assumption that we have a Machine Learning model in production, the project
aims to check in regular intervals (per crontab configuration) for new datasets
and acts upon any new data that is saved. 
The model is tested for checking whether model drift, re-train if needed 
and new reporting of the model performance, data quality and timing of execution.

## Context
The project aims at building a classifier for the prediction of customer churn and overall risk assessment.
Synthetic datasets are provided to train and simulate continuous operations. The focus is on the MLOps components rather than the quality of the model.

## Workflow overview
The workflow is broken down into the following components which are all separated:
- data ingestion
- model training (logisticRegression sklearn model for binary classification)
- model scoring
- deployment of pipeline into production
- model monitoring, reporting and statistics - API set-up for ML diagnostics and results
- process automation with data ingestion and model drift detection using CRON job
- retraining / redeployment in case of model drift

# How to use
The project is deployed under Mac python WSL2 linux. Deployed API using Flask framework.
The model API should be launched before executing project components. This can be achieved by running app.py script instantiating multiple project API endpoints including inference capability. Other components of the project include:
- ingestion.py to ingest data and prepare model training
- training.py to train a logistic regression model
- scoring.py to score the model in production against a test dataset
- deployment.py to deploy key artifacts(trained model) to production
- diagnostics.py gather various analysis and diagnostics
- apicalls.py calls all diagnostics through the API and generate a consolidated report
- reporting.py allows to generate a full pdf report gathering performance plots, metrics and other useful statistics
- fullprocess.py should be run regularly using a CRON job. It monitors new data availability, checks model drift, decides to retrain and redeploy an updated model in case shifting is detected.

## Cron job implementation
- activate cron jobs in WSL2 using sudo service cron start (if not already active)
- create a new cron job using crontab -e
- the cron job should run the fullprocess.py script every 10 minutes in order to automate the whole process from data ingestion to model redeployment as needed









