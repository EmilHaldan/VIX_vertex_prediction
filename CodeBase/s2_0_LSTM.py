from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Layer, Concatenate, Dropout, LSTM, Attention, MultiHeadAttention
from keras.layers import Input, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import random as rn
from collections import OrderedDict
import time
from datetime import datetime
import shutil

from s1_0_Dataloader import load_lstm_data, get_window_and_feature_space



def build_LSTM_model(window_size, feature_space, input_LSTM_units, hidden_LSTM_units, 
                        dense_units, loss_func, opt, drop_out_rate, learning_rate, specific_model_name):
    """
    window_size : a variable dictating how large the window for a sample should be
    input_LSTM_units : the number of LSTM units in the input layer
    feature_space : the number of features in the input layer
    hidden_LSTM_units : a list with the the number of LSTM units in the hidden layer. If the list is empty, no hidden layers are created.
    dense_units : a list with the the numbers of dense units in the hidden layer. If the list is empty, no dense layers are created.
    loss_func : the loss function to be used in the model. Is a loss function object from keras.losses
    opt : the optimizer to be used in the model. Is a optimizer object from keras.optimizers
    drop_out_rate : the dropout rate to be used in the model. Is a float between 0 and 1
    specific_model_name : the id of the model iteration. Is a string

    Output:
    model : the compiled Neural network model
    """

    model = Sequential()        

    amount_of_lstm_hidden_layers = len(hidden_LSTM_units)
    if amount_of_lstm_hidden_layers == 0:
        return_sequences_bool = False
    else: 
        return_sequences_bool = True

    model.add(LSTM(units= input_LSTM_units, 
                   input_shape=(window_size, feature_space, ), return_sequences=return_sequences_bool,
                   name = "input_layer_LSTM__{}".format(specific_model_name)))
    
    # Hidden layers
    for i in range(amount_of_lstm_hidden_layers):
        if amount_of_lstm_hidden_layers-1 == i:
            return_sequences_bool = False
        else:
            return_sequences_bool = True
        model.add(keras.layers.BatchNormalization(name = "{}_h_layer_BatchNorm_id{}".format(i+1,specific_model_name))) # removed with the hypothesis that LSTM already normalize with sigmoid functions.
        model.add(LSTM(units= hidden_LSTM_units[i] , return_sequences=return_sequences_bool, 
                       dropout = drop_out_rate, name = "{}_h_layer_LSTM__{}"   .format(i+1,specific_model_name)))

    for units_idx, units in enumerate(dense_units):
        model.add(Dense(units, name = "{}_dense_layer__{}".format(units_idx,specific_model_name) , activation='elu'))
        model.add(Dropout(drop_out_rate, name = "{}_dense_layer_dropout__{}".format(units_idx,specific_model_name)))

    model.add(Dense(2, name = "output_regression__{}".format(specific_model_name) ))

    opt.learning_rate = learning_rate
    model.compile(optimizer= opt, loss= loss_func, metrics=['MAE','MSE'])

    print("Model summary before training:")
    print(model.summary())
    print("")

    return model


def save_model_and_pred(y_pred, val_y_pred, y_test, y_val, val_dates, test_dates, model, model_name, specific_model_path):
    """
    y_pred : the predictions on the test data
    val_y_pred : the predictions on the validation data
    model : the trained Neural network model
    model_name : the name of the model general model. aka. the filename. Is a string
    specific_model_path : the path to the specific model

    Output:
    None
    """

    model.save(f"{specific_model_path}/{model_name}/{model_name}.h5")

    high_y_target = [x for x in y_test[:,0]]
    high_y_pred   = [x[0] for x in y_pred]
    low_y_target  = [x for x in y_test[:,1]]
    low_y_pred    = [x[1] for x in y_pred]

    y_pred_df = pd.DataFrame({"date": test_dates, 
                            "high_y_target": high_y_target , "high_y_pred": high_y_pred, 
                            "low_y_target" :  low_y_target , "low_y_pred" : low_y_pred})

    pred_high_y_target = [x for x in y_val[:,0]]
    pred_high_y_pred   = [x[0] for x in val_y_pred]
    pred_low_y_target  = [x for x in y_val[:,1]]
    pred_low_y_pred    = [x[1] for x in val_y_pred]

    val_y_pred_df = pd.DataFrame({"date": val_dates, 
                            "high_y_target": pred_high_y_target, "high_y_pred": pred_high_y_pred, 
                            "low_y_target" : pred_low_y_target,  "low_y_pred" : pred_low_y_pred })

    y_pred_df.to_csv(f"{specific_model_path}/{model_name}/{model_name}_y_pred.csv", index=False)
    val_y_pred_df.to_csv(f"{specific_model_path}/{model_name}/{model_name}_val_y_pred.csv", index=False)


def save_metrics(test_loss, test_mae, test_mse, val_loss, val_mae, val_mse, epochs_trained, specific_model_path, model_name):
    metrics_df = pd.DataFrame({"test_loss": [test_loss], "test_mae": [test_mae], "test_mse": [test_mse], 
                               "val_loss": [val_loss], "val_mae": [val_mae], "val_mse": [val_mse], 
                               "epochs_trained": [epochs_trained]})

    metrics_df.to_csv(f"{specific_model_path}/{model_name}/{model_name}_metrics.csv", index=False)

def check_or_create_folder(specific_model_path):
    if not os.path.isdir(specific_model_path):
        os.mkdir(specific_model_path)


def check_amount_of_models(specific_model_path):
    print("DONE TRAINING MODELS")
    return len(os.listdir(specific_model_path)) >= 100



def find_model_name(specific_model_path):
    check_or_create_folder(specific_model_path)
    num_files_req = 5

    for folder in os.listdir(specific_model_path):
        if os.path.isdir(f"{specific_model_path}/{folder}"):
            if len(os.listdir(f"{specific_model_path}/{folder}")) != num_files_req:
                print("\n"*5)
                print("WARNING!    "*10)
                print("WARNING!    "*10)
                print("WARNING!    "*10)
                print(f"REMOVING FOLDER:", folder, f"as it does not contain {num_files_req} files")
                print("WARNING!    "*10)
                print("WARNING!    "*10)
                print("WARNING!    "*10)
                print("\n"*5)
                time.sleep(2)
                shutil.rmtree(f"{specific_model_path}/{folder}")

    highest_available_number = 0
    for i in range(1,1000):
        highest_available_number = i
        if os.path.isdir(f"{specific_model_path}/LSTM_{i}"):
            continue
        else:
            break

    os.mkdir(f"{specific_model_path}/LSTM_{highest_available_number}")

    return f"LSTM_{highest_available_number}"


def train_model(model, epochs, patience, batch_size, 
                X_train, X_val, X_test, y_train, y_val, y_test, 
                train_dates, val_dates, test_dates ,
                specific_model_path, model_name):
    """
    model : the compiled Neural network model
    epochs : the maximum number of epochs to train the model
    patience : the number of epochs to wait before early stopping
    batch_size : the batch size to be used in the model
    train_X : the training data
    train_y : the training labels
    test_X : the test data
    test_y : the test labels
    specific_model_name : the name of the specific model being trained. Is a string
    model_name : the name of the model general model. aka. the filename. Is a string

    Output:
    model : the trained Neural network model
    """

    # setting the callbacks
    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience , restore_best_weights = True)
    csv_logger = CSVLogger(f"{specific_model_path}/{model_name}/training_log.csv", append=True, separator=';')

    cb_list = [es_callback, csv_logger]

    model_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=cb_list)

    test_loss,test_mae,test_mse = model.evaluate(X_test, y_test, verbose=1)
    val_loss, val_mae, val_mse  = model.evaluate(X_val,  y_val,  verbose=1)

    epochs_trained = len(model_history.history['loss']) - patience

    val_y_pred = [y for y in model.predict(X_val)]
    y_pred = [y for y in model.predict(X_test)]

    save_model_and_pred(y_pred = y_pred, val_y_pred = val_y_pred, 
                        y_test = y_test, y_val = y_val, 
                        val_dates = val_dates, test_dates = test_dates,
                        model = model, model_name = model_name, 
                        specific_model_path = specific_model_path)

    save_metrics(test_loss, test_mae, test_mse, val_loss, val_mae, val_mse, epochs_trained, specific_model_path, model_name)
           
    return model, test_loss,test_mae,test_mse, val_loss, val_mae, val_mse, y_pred, val_y_pred, epochs_trained


if __name__ == "__main__":
    
    for i in range(1, 100 ):
        
        # Hyper Parameters
        input_LSTM_units = 256
        hidden_LSTM_units = [256]
        dense_units = [256]
        loss_func = keras.losses.MeanSquaredError()
        opt = keras.optimizers.Adamax()
        drop_out_rate = 0.1
        specific_model_name = "LSTM_all_features_MSE"
        learning_rate = 0.00001
        epochs = 500
        patience = 15
        batch_size = 32
        specific_model_path = f"ML_saved_models/{specific_model_name}"
        model_name = find_model_name(specific_model_path)

        window_size = 20
        X_train, y_train, train_dates = load_lstm_data(window_size = window_size, range_type = "train", version = 4, verbose = False)
        X_val, y_val, val_dates       = load_lstm_data(window_size = window_size, range_type = "val", version = 4, verbose = False)
        X_test, y_test, test_dates    = load_lstm_data(window_size = window_size, range_type = "test", version = 4, verbose = False)
        window_size, feature_space, target_length = get_window_and_feature_space(X_train, y_train)

        if check_amount_of_models(specific_model_path):
            break

        model = build_LSTM_model(window_size, feature_space, input_LSTM_units, hidden_LSTM_units, 
                            dense_units, loss_func, opt, drop_out_rate, learning_rate, specific_model_name)

        model, test_loss,test_mae,test_mse, val_loss, val_mae, val_mse, y_pred, val_y_pred, epochs_trained = train_model(
                                                                                    model = model, epochs = epochs, 
                                                                                    patience = patience, batch_size = batch_size, 
                                                                                    X_train = X_train, X_val = X_val, X_test = X_test, 
                                                                                    y_train = y_train, y_val = y_val, y_test = y_test, 
                                                                                    train_dates = train_dates, val_dates = val_dates, test_dates = test_dates ,
                                                                                    specific_model_path = specific_model_path, 
                                                                                    model_name = model_name)

        print("test_loss:", test_loss)
        print("test_mae:", test_mae)
        print("test_mse:", test_mse)
        print("val_loss:", val_loss)
        print("val_mae:", val_mae)
        print("val_mse:", val_mse)
        print("epochs_trained:", epochs_trained)
        print("")    