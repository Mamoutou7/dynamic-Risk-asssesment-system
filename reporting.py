import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os

from diagnostics import model_predictions

###############Load config.json and get path variables

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    dataset_path = os.path.join(test_data_path, 'testdata.csv')
    dataset = pd.read_csv(dataset_path)

    # prediction performing
    predicted = model_predictions()

    # getting target
    y = dataset['exited']
    # calculate confusion matrix
    cm = metrics.confusion_matrix(y, predicted)

    # Setting default size of the plot
    # Setting default fontsize used in the plot
    plt.rcParams['figure.figsize'] = (10.0, 9.0)
    plt.rcParams['font.size'] = 20

    # Implementing visualization of Confusion Matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

    # Plotting Confusion Matrix
    # Setting colour map to be used
    cm_display.plot(cmap='OrRd', xticks_rotation=25)

    # Setting fontsize for xticks and yticks
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Giving name to the plot
    plt.title('Confusion Matrix', fontsize=24)

    # write the confusion matrix to the workspace
    save_path = os.path.join(model_path, 'confusionmatrix.png')
    plt.savefig(save_path)



if __name__ == '__main__':
    score_model()
