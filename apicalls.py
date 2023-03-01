import os
import json
import requests
import logging

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/apicalls.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])

test_file_path = os.path.join(test_data_path, 'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.get(URL + '/prediction' + f'?filename={test_file_path}').content
response2 = requests.get(URL + '/scoring').content
response3 = requests.get(URL + '/summarystats').content
response4 = requests.get(URL + '/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

logging.info("Writing the report in apireturns.txt")
# write the responses to your workspace
output_file_path = os.path.join(model_path, 'apireturns.txt')
with open(output_file_path, 'w') as file:
    file.write(str(responses))




