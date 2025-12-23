import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def build_lstm_regressor_advanced(
    input_shape,
    num_classes,
    lstm_units=[64, 32],            # 각 LSTM layer의 유닛 수
    dense_units=[32],              # Dense layer 유닛 수
    dropout_rate=0.3,
    use_batchnorm=False,
    l2_reg=0.0,
    learning_rate=1e-4
):
    model = Sequential()
    # LSTM Layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(LSTM(units, return_sequences=return_seq,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Dense Layers
    for units in dense_units:
        model.add(Dense(units, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(num_classes, activation='sigmoid'))
    # Loss
    loss_function = 'mse'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae']
    )
    
    return model

def build_lstm_classifier_advanced(
    input_shape,
    num_classes,
    oh=False,
    lstm_units=[64, 32],            # 각 LSTM layer의 유닛 수
    dense_units=[32],              # Dense layer 유닛 수
    dropout_rate=0.3,
    use_batchnorm=False,
    l2_reg=0.0,
    learning_rate=1e-4
):
    model = Sequential()
    
    # LSTM Layers
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq,
                           input_shape=input_shape,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        else:
            model.add(LSTM(units, return_sequences=return_seq,
                           kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Dense Layers
    for units in dense_units:
        model.add(Dense(units, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg)))
        if use_batchnorm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Loss
    loss_function = 'categorical_crossentropy' if oh else 'sparse_categorical_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )
    
    return model