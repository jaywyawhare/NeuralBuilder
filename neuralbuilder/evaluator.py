# neuralbuilder/evaluator.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def model_eval(y_test, y_pred, task_type):
    """
    Evaluate the model performance using various metrics and plot a confusion matrix or scatter plot.

    Args:
        y_test (numpy array): True labels of the test data.
        y_pred (numpy array): Predicted labels of the test data.
        task_type (str): The type of task. Supported values are 'classification' and 'regression'.

    Returns:
        None
    """

    if task_type == 'classification':
        y_test = np.argmax(y_test, axis=1)

        xtick = np.unique(y_test)
        ytick = np.unique(y_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Accuracy\t:\t{accuracy}')
        print(f'Precision\t:\t{precision}')
        print(f'Recall\t:\t{recall}')
        print(f'F1 score\t:\t{f1}')

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=xtick, yticklabels=ytick)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    elif task_type == 'regression':
        plt.scatter(y_test, y_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

    else:
        raise ValueError(f"Unsupported task type: {task_type}")
