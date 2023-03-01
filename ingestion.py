import pandas as pd
import os
import json
import logging

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/ingestion.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

#############Load config.json and get input and output paths
logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

current_dir = os.getcwd()

#############Function for data ingestion
def merge_multiple_dataframe():
    """
    Combine the data in all of individual datasets into a single DataFrame and save on the disk.
    :return: Single DataFrame recorded and save on the disk
    """
    final_df = pd.DataFrame(
            columns=['corporation',
                     'lastmonth_activity',
                     'lastyear_activity',
                     'number_of_employees',
                     'exited']
    )
    # list of files
    filenames = os.listdir(input_folder_path)
    # list to content individual file
    ingested_files = []
    # check for datasets, compile them together, and write to an output file
    for each_filename in filenames:
        each_filename_path = os.path.join(input_folder_path, each_filename)
        current_df = pd.read_csv(each_filename_path)
        final_df = pd.concat([final_df, current_df], axis=0)
        ingested_files.append(each_filename)

    # drop duplicates
    result = final_df.drop_duplicates()
    # path to store the final data
    data_path = os.path.join(output_folder_path, 'finaldata.csv') # f'{today}_final_data.csv'
    # save the final data in csv
    result.to_csv(data_path, index=False)

    data_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    # list of the ingested .csv files
    with open(data_path, 'w') as f:
        f.write(str(ingested_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
