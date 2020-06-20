import os
from time import time, process_time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from utility import *
from LSTM import *

homedir = get_homedir()

TODAY = '0526' # Date of today. Only used in defining PATH_PREP variable(= which folder to search for data)

PATH_PREP = f"{homedir}/JK/preprocessing/{TODAY}" # Which folder to search for the preprocessed data.
PATH = f"{homedir}/JK/prediction/{TODAY}_test"         # All outputs will be saved in this folder.

with open(PATH_PREP+f'/date_ed.txt', 'r') as f:
    date_ed = pd.Timestamp(f.read()) # Last date in the preprocessed data. 
                                     # Same as the final date to be trained in the previous cell.
timedelta = 0 # How many dates from date_ed not to be included in training.
              # Only necessary for using custom training timeline.
split_ratio = None              # Training-validation splitting ratio
QUANTILE = list(quantileList)   
history_size = 7                # Size of history window
target_size = 14                # Size of target window
step_size = 1                   
NUM_CELLS = 128                 # Number of cells in LSTM layer
lr = 0.001                      # Learning rate
dp = 0.2                        # Dropout rate
EPOCHS = 100                    # Number of epochs for training
#######################################################################################
"""
Load necessary data from PATH_PREP.
"""
with open(PATH_PREP+f'/FIPS.txt', 'r') as f:
    FIPS_total = eval(f.read())

data_ctg = np.load(PATH_PREP+f'/data_ctg.npy', allow_pickle=True)
print(f'Categorical data of shape {data_ctg.shape} is loaded.')
data_ts = np.load(PATH_PREP+f'/data_ts.npy', allow_pickle=True)
print(f'Timeseries data of shape {data_ts.shape} is loaded.')
data_ts = data_ts[:, :-timedelta, :]

with open(PATH_PREP+f'/columns_ctg.txt', 'r') as f:
    columns_ctg = eval(f.read())
with open(PATH_PREP+f'/columns_ts.txt', 'r') as f:
    columns_ts = eval(f.read())
print(f'# of features = {len(columns_ctg)+len(columns_ts)}')

target_idx = columns_ts.index('deaths')
print('target_idx:', target_idx)

try:
    os.mkdir(PATH)
except OSError as error:
    print(error)

"""
Generate the training data, an instance of tensorflow.Dataset class.
"""
# X_train, y_train, X_val, y_val, C_train, C_val = train_val_split(data_ts, data_ctg, target_idx, history_size, target_size, split_ratio=split_ratio, step_size=step_size)
X_train, y_train, C_train = train_full(data_ts, data_ctg, target_idx, history_size, target_size, step_size=step_size)

scaler_ts, scaler_ctg = get_StandardScaler(X_train, C_train)

X_train, y_train = normalizer(scaler_ts, X_train, y_train, target_idx)
# X_val, y_val = normalizer(scaler_ts, X_val, y_val, target_idx)
C_train = normalizer(scaler_ctg, C_train)
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
for i in range(3):
    model, history = LSTM_fit_mult(train_data, lr=lr, NUM_CELLS=NUM_CELLS, EPOCHS=EPOCHS, dp=dp, monitor=True, earlystop=False, verbose=2)
    FILEPATH = f"/LSTM_mult_hist_size_{history_size}"
    plot_train_history(history, title=f'History size={history_size}, dropout={dp}', path=PATH+FILEPATH+f'_history_{i}.png')
    model.save_weights(PATH+FILEPATH+f'_weights', save_format="tf")

    df_future = predict_future_mult(model, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=FIPS_total, date_ed=date_ed-pd.Timedelta(days=timedelta))
    df_future.to_csv(PATH+f'/LSTM_mult_hist_size_{history_size}_{TODAY}_{i}.csv', index=False)

    model_test = SingleLayerConditionalRNN(NUM_CELLS, target_size, dp, quantileList)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model_test.compile(optimizer=optimizer, loss=lambda y_p, y: MultiQuantileLoss(quantileList, target_size, y_p, y))
    load_status = model_test.load_weights(PATH+FILEPATH+f'_weights')
    # print(load_status.assert_consumed())
    df_future_test = predict_future_mult(model_test, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=FIPS_total, date_ed=date_ed-pd.Timedelta(days=timedelta))
    df_future_test.to_csv(PATH+f'/LSTM_mult_hist_size_{history_size}_{TODAY}_{i}_test.csv', index=False)