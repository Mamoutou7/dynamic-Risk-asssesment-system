import shutil
import os
import json
import logging

# Initialising logger for checking steps

logging.basicConfig(
    filename='./logs/deployment.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and correct path variable

logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    if not os.path.exists(prod_deployment_path):
        logging.info("Creating production deployment directory")
        os.makedirs(prod_deployment_path)

    logging.info("Copying all files to production deployment path")
    files_to_copy = ['latestscore.txt', 'trainedmodel.pkl']
    for file in files_to_copy:
        shutil.copy(os.path.join(output_model_path, file), os.path.join(prod_deployment_path, file))

    # copy ingestion data file (ingestfiles.txt)
    shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), os.path.join(prod_deployment_path, 'ingestedfiles.txt'))

if __name__ == '__main__':
    store_model_into_pickle()
