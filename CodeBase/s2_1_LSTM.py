from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Layer, Concatenate, Dropout, LSTM, Attention, MultiHeadAttention
from keras.layers import Input, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras import backend as K
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
from s2_0_LSTM import *


def custom_loss(y_true, y_pred):
    high_true, low_true = y_true[:, 0], y_true[:, 1]
    high_pred, low_pred = y_pred[:, 0], y_pred[:, 1]

    mse_high = K.square(high_pred - high_true)
    mse_low = K.square(low_pred - low_true)

    # The penalty increases exponentially with the value of the target and prediction
    # The division by 2 was made in order for the visualization to be clearer, while leaving the exponential functionality intact
    # This turned out to be completely redundant as the plans to visualize the loss function with the same color bar were abandoned
    penalty_high = mse_high * K.exp(high_true) * K.exp(high_pred) /2
    penalty_low = mse_low * K.exp(low_true) * K.exp(low_pred) /2

    # Combining penalties and compute overall loss
    total_loss = K.mean(penalty_high + penalty_low, axis=-1)

    return total_loss



if __name__ == "__main__":
    
    for i in range(1, 100):
        
        # Hyper Parameters
        input_LSTM_units = 256
        hidden_LSTM_units = [256]
        dense_units = [256]
        loss_func = custom_loss
        opt = keras.optimizers.Adamax()
        drop_out_rate = 0.1
        specific_model_name = "LSTM_all_features_custom_loss"
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