import os
import sys
import json
from time import time, process_time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from misc.utility import *
from LSTM.LSTM import *

homedir = get_homedir()
logger = logging.getLogger('main.trainer')

def LSTM_trainer(config_name, tmp):
    PATH_PREP = os.path.join(homedir, 'LSTM/preprocessing', tmp)
    logger.info(f"Using preprocessed data in {PATH_PREP}")

    with open(os.path.join(PATH_PREP, 'config.json')) as f:
        config_dict = json.load(f)

    config_st = pd.Timestamp(config_dict['start_date'])
    if config_dict['end_date'] is None:
        config_ed = config_st + pd.Timedelta(days=13)
    else:
        config_ed = pd.Timestamp(config_dict['end_date'])
    with open(config_name, 'r') as f:
        temp_dict = json.load(f)
        config_out = temp_dict["out_files"]

    tmp = str(round(1000*pd.Timestamp.utcnow().timestamp()))

    PATH = os.path.join(homedir, 'LSTM', 'prediction', tmp)         # All outputs will be saved in this folder.

    with open(os.path.join(PATH_PREP, 'date_ed.txt'), 'r') as f:
        date_ed = pd.Timestamp(f.read()) # Last date in the preprocessed data. 
                                        # Same as the final date to be trained in the previous cell.
    timedelta = max(0, (date_ed - config_st).days + 1) # How many dates from date_ed not to be included in training.
                # Only necessary for using custom training timeline.
    split_ratio = None              # Training-validation splitting ratio
    QUANTILE = list(quantileList)   
    hparam = {
        "history_size": 7,          # Size of history window
        "NUM_CELLS": 128,           # Number of cells in LSTM layer
        "lr": 0.001,                # Learning rate
        "dp_ctg": 0.2,              # Dropout rate(categorical inputs)
        "dp_ts" : 0.2,              # Dropout rate(timeseries inputs)
        "EPOCHS": 15                # Number of epochs for training
    }
    target_size = (config_ed-date_ed).days                # Size of target window
    step_size = 1                   
    #######################################################################################
    """
    Load necessary data from PATH_PREP.
    """
    with open(os.path.join(PATH_PREP, 'FIPS.txt'), 'r') as f:
        FIPS_total = eval(f.read())

    data_ctg = np.load(os.path.join(PATH_PREP, 'data_ctg.npy'), allow_pickle=True)
    logger.info(f'Categorical data of shape {data_ctg.shape} is loaded.')
    data_ts = np.load(os.path.join(PATH_PREP, 'data_ts.npy'), allow_pickle=True)
    logger.info(f'Timeseries data of shape {data_ts.shape} is loaded.')
    if timedelta>0:
        data_ts = data_ts[:, :-timedelta, :]

    with open(os.path.join(PATH_PREP, 'columns_ctg.txt'), 'r') as f:
        columns_ctg = eval(f.read())
    with open(os.path.join(PATH_PREP, 'columns_ts.txt'), 'r') as f:
        columns_ts = eval(f.read())
    logger.info(f'# of features = {len(columns_ctg)+len(columns_ts)}')

    target_idx = columns_ts.index('deaths')
    logger.info(f'target_idx: {target_idx}')

    os.makedirs(PATH, mode=0o770, exist_ok=True)

    """
    Generate the training data, an instance of tensorflow.Dataset class.
    """
    X_train, C_train, y_train = train_full(data_ts, data_ctg, target_idx, hparam["history_size"], target_size, step_size=step_size)
    # X_train, y_train, X_val, y_val, C_train, C_val = train_val_split(data_ts, data_ctg, target_idx, history_size, target_size, split_ratio=split_ratio, step_size=step_size)

    scaler_ts, scaler_ctg = get_StandardScaler(X_train, C_train)
    mu, sigma = scaler_ts.mean_[target_idx], scaler_ts.scale_[target_idx]
    logger.debug('mu={mu}, sigma={sigma}')

    """
    Z-score input data.
    """
    X_train = normalizer(scaler_ts, X_train)
    C_train = normalizer(scaler_ctg, C_train)
    # X_val, y_val = normalizer(scaler_ts, X_val, y_val, target_idx)
    # C_train, C_val = normalizer(scaler_ctg, C_train), normalizer(scaler_ctg, C_val)

    # train_data, val_data = load_Dataset(X_train, C_train, y_train, X_val, C_val, y_val)
    train_data = load_Dataset(X_train, C_train, y_train)

    # model, history = LSTM_fit_mult(train_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp, monitor=True, earlystop=False, verbose=2)
    # FILEPATH = f"/LSTM_mult_hist_size_{history_size}"
    # plot_train_history(history, title=f'History size={history_size}, dropout={dp}', path=PATH+FILEPATH+'_history.png')

    # df_future = predict_future_mult(model, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=FIPS_total, date_ed=date_ed)
    # df_future.to_csv(PATH+f'/LSTM_{TODAY}.csv', index=False)
    """
    Train the model and forecast.
    """
    # callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, baseline=0.1)]
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(PATH, "log"))]

    model, history = LSTM_fit_mult(train_data, hparam=hparam, monitor=True, callbacks=callbacks, verbose=2, mu=mu, sigma=sigma, test=False)
    # Plot 
    plot_train_history(history, title=f'History size={hparam["history_size"]}', path=os.path.join(PATH, 'history.png'))
    model.save_weights(os.path.join(PATH, 'weights'), save_format="tf")

    df_future = predict_future_mult(model, data_ts, data_ctg, scaler_ts, scaler_ctg, hparam['history_size'],
                                    target_idx, FIPS=FIPS_total, date_ed=date_ed-pd.Timedelta(days=timedelta))
    df_future.to_csv(os.path.join(PATH, 'LSTM_mult.csv'), index=False)

    # model_test = SingleLayerConditionalRNN(NUM_CELLS, target_size, dp, quantileList)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # model_test.compile(optimizer=optimizer, loss=lambda y_p, y: MultiQuantileLoss(quantileList, target_size, y_p, y))
    # load_status = model_test.load_weights(PATH+FILEPATH+f'_weights')
    # # print(load_status.assert_consumed())
    # df_future_test = predict_future_mult(model_test, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=FIPS_total, date_ed=date_ed-pd.Timedelta(days=timedelta))
    # df_future_test.to_csv(PATH+f'/LSTM_mult_hist_size_{history_size}_{TODAY}_test.csv', index=False)

    """
    Forecast into submission.
    """
    logger.info('Save into submission format.')
    df_submission = prediction_to_submission([df_future], base=os.path.join(PATH_PREP, 'sample_submission.csv'))
    df_submission.to_csv(config_out, index=False)