# neuralbuilder/plotter.py

import matplotlib.pyplot as plt

def plot_epoch(history, y_test, y_pred, task_type):
    '''
    Plot accuracy and loss values during training and validation as a function of epochs.

    Args:
        history (keras.callbacks.History): Object returned by fit() containing information about the training history.
        y_test (numpy array): True labels of the test data.
        y_pred (numpy array): Predicted labels of the test data.
        task_type (str): The type of task. Supported values are 'classification' and 'regression'.

    Returns:
        None

    Displays two plots:
    1. Accuracy vs Epochs or Loss vs Epochs: shows the training and validation accuracy/loss values as a function of epochs.
    '''

    if task_type == 'classification':
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs Graph')
        plt.legend()
        plt.show()

    elif task_type == 'regression':
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs Graph')
        plt.legend()
        plt.show()

    else:
        raise ValueError(f"Unsupported task type: {task_type}")
