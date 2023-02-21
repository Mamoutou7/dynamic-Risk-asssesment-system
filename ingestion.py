import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

current_dir = os.getcwd()

# get current date
today = datetime.today().strftime('%Y%m%d')

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
    data_path = os.path.join(output_folder_path, f'{today}_final_data.csv')
    # save the final data in csv
    result.to_csv(data_path, index=False)

    data_path = os.path.join(output_folder_path, f'{today}_ingestedfiles.txt')
    # list of the ingested .csv files
    with open(data_path, 'w') as f:
        f.write(str(ingested_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
