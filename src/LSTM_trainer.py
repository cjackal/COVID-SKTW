import os
import sys
import json
from time import time, process_time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from .misc.utility import get_homedir, reformat_prediction
from .LSTM import *

homedir = get_homedir()
logger = logging.getLogger('main.trainer')

def LSTM_trainer(config_name, tmp, ver='frozen'):
    PATH_PREP = os.path.join(homedir, 'tmp', 'preprocessing', tmp)
    logger.info(f"Using preprocessed data in {PATH_PREP}")

    with open(config_name, 'r') as f:
        config_dict = json.load(f)
    hparam = {
        "history_size": 7,          # Size of history window
        "NUM_CELLS": 128,           # Number of cells in LSTM layer
        "lr": 0.001,                # Learning rate
        "dp_ctg": 0.2,              # Dropout rate(categorical inputs)
        "dp_ts" : 0.2,              # Dropout rate(timeseries inputs)
        "EPOCHS": 15,               # Number of epochs for training
    }
    if 'hparam' in config_dict:
        for key in config_dict['hparam']:
            hparam[key] = config_dict['hparam'][key]

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
    os.makedirs(os.path.dirname(config_out), mode=0o770, exist_ok=True)

    tmp = str(round(1000*pd.Timestamp.utcnow().timestamp()))
    # Directory to save temporary files.
    PATH = os.path.join(homedir, 'tmp', 'prediction', tmp)
    os.makedirs(PATH, mode=0o770, exist_ok=True)

    with open(os.path.join(PATH_PREP, 'date_ed.txt'), 'r') as f:
        # Last date in the preprocessed data.
        # Same as the final date to be trained in the previous cell.
        date_ed = pd.Timestamp(f.read())

    # How many dates from date_ed not to be included in training.
    # Only necessary for non-maximal training timeline.
    timedelta = max(0, (date_ed - config_st).days + 1)

    split_ratio = None              # Training-validation splitting ratio
    target_size = (config_ed - date_ed).days             # Size of target window
    step_size = 1
    ############################################################################
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

    """
    Generate the training data, an instance of tensorflow.Dataset class.
    """
    X_train, C_train, y_train = train_full(data_ts, data_ctg, target_idx, hparam["history_size"], target_size, step_size=step_size)

    scaler_ts, scaler_ctg = get_StandardScaler(X_train, C_train)
    mu, sigma = scaler_ts.mean_[target_idx], scaler_ts.scale_[target_idx]
    logger.debug(f'mu={mu}, sigma={sigma}')

    """
    Z-score input data.
    """
    X_train = normalizer(scaler_ts, X_train)
    C_train = normalizer(scaler_ctg, C_train)

    train_data = load_Dataset(X_train, C_train, y_train)

    """
    Train the model and forecast.
    """
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(PATH, "log"))]

    model, history = LSTM_fit_mult(train_data, hparam=hparam, monitor=True,
                                    callbacks=callbacks, verbose=2, mu=mu,
                                    sigma=sigma, ver=ver)

    # Plot learning curve.
    plot_train_history(history, title=f'History size={hparam["history_size"]}',
                        path=os.path.join(PATH, 'history.png'))
    model.save_weights(os.path.join(PATH, 'weights'), save_format="tf")

    df_future = predict_future_mult(model, data_ts, data_ctg, scaler_ts,
                                scaler_ctg, hparam['history_size'], target_idx,
                                FIPS=FIPS_total, date_ed=date_ed-pd.Timedelta(days=timedelta))
    df_future.to_csv(os.path.join(PATH, 'LSTM_mult.csv'), index=False)

    with open(os.path.join(PATH, 'date_ed.txt'), 'w') as f:
        print(date_ed.strftime('%Y-%m-%d'), file=f)

    """
    Reformat into single-index form.
    """
    logger.info('Save into single-index format.')
    df_submission = reformat_prediction([df_future], base=os.path.join(PATH_PREP, 'frame.csv'))
    df_submission.to_csv(config_out, index=False)