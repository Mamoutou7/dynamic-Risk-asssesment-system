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

    :return:
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
    # path to store the final data
    final_data = os.path.join(output_folder_path, f'{today}_final_data.csv')

    # check for datasets, compile them together, and write to an output file
    for each_filename in filenames:
        each_filename_path = os.path.join(input_folder_path, each_filename)
        current_df = pd.read_csv(each_filename_path)
        final_df = pd.concat([final_df, current_df], axis=0)

    # drop duplicates
    result = final_df.drop_duplicates()
    # save the final data in csv
    result.to_csv(final_data, index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
