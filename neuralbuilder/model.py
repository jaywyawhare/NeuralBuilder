# neuralbuilder/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

def createModel(model_type, num_layers, layer_types, layer_units, input_shape, task_type, output_units=None):
    '''
    Create a deep learning model with specified architecture for classification or regression tasks.

    Args:
        model_type (str): The type of model to create. Supported values are 'Sequentials', 'LSTM', and 'Convolutional'.
        num_layers (int): The number of layers to include in the model.
        layer_types (list[str]): A list of layer types to include in the model.
        layer_units (list[int]): A list of the number of units for each layer in the model.
        input_shape (tuple[int]): The shape of the input data.
        task_type (str): The type of task. Supported values are 'classification' and 'regression'.
        output_units (int, optional): The number of output units in the final layer of the model.
                                      Defaults to None.

    Returns:
        tensorflow.keras.models.Model: The compiled deep learning model.

    Raises:
        ValueError: If an unsupported model type or task type is specified.
    '''

    model = Sequential()
    if model_type == 'Sequentials':
        for i in range(num_layers):
            if layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")
    
    elif model_type == 'LSTM':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'LSTM':
                model.add(LSTM(layer_units[i], input_shape=input_shape))
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")

    elif model_type == 'Convolutional1D':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'Conv1D':
                model.add(Conv1D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling1D':
                model.add(MaxPooling1D(pool_size=2))
            elif layer_types[i] == 'Conv2D':
                model.add(Conv2D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling2D':
                model.add(MaxPooling2D(pool_size=2))
            elif layer_types[i] == 'Flatten':
                model.add(Flatten())
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")

    elif model_type == 'Convolutional2D':
        model.add(Reshape((input_shape[1], 1)))
        for i in range(num_layers):
            if layer_types[i] == 'Conv2D':
                model.add(Conv1D(filters=layer_units[i], kernel_size=3, activation='relu', input_shape=input_shape))
            elif layer_types[i] == 'MaxPooling2D':
                model.add(MaxPooling1D(pool_size=2))
            elif layer_types[i] == 'Flatten':
                model.add(Flatten())
            elif layer_types[i] == 'Dense':
                model.add(Dense(layer_units[i], activation='relu'))
            elif layer_types[i] == 'Dropout':
                model.add(Dropout(layer_units[i]))
            else:
                raise ValueError(f"Unsupported layer type: {layer_types[i]}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if task_type == 'classification':
        activation = 'softmax'
        if output_units is None:
            raise ValueError("The number of output units must be specified for classification.")
    elif task_type == 'regression':
        activation = 'linear'
        if output_units is None:
            output_units = 1
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    model.add(Dense(output_units, activation=activation))
    model.compile(loss='categorical_crossentropy' if task_type == 'classification' else 'mean_squared_error',
                  optimizer=Adam(0.001),
                  metrics=['accuracy' if task_type == 'classification' else 'mse'])
    model.build(input_shape=input_shape)
    model.summary()
    return model
