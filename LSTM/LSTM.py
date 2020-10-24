import platform
import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

_quantileList = np.linspace(0.1, 0.9, 9)
if platform.system()=='Linux': ### In a session
    import matplotlib
    matplotlib.use('Agg')

logger = logging.getLogger("main.trainer.LSTM")

class LRFinder(tf.keras.callbacks.Callback):
    """Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 10 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self, path=None):
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        if path is None:
            fig.show()
        else:
            fig.savefig(path)

class ConditionalRNN(tf.keras.layers.Layer):
    """
    Custom conditional RNN layer.
    Credit to Philippe Rémy(https://github.com/philipperemy/cond_rnn.git)
    """
    # Arguments to the RNN like return_sequences, return_state...
    def __init__(self, units, cell=tf.keras.layers.LSTMCell, categorical_dropout=0.0, timeseries_dropout=0.0,
                *args, **kwargs):
        """
        Conditional RNN. Conditions time series on categorical data.
        :param units: int, The number of units in the RNN Cell
        :param cell: string, cell class or object (pre-instantiated). In the case of string, 'GRU',
        'LSTM' and 'RNN' are supported.
        :param args: Any parameters of the tf.keras.layers.RNN class, such as return_sequences,
        return_state, stateful, unroll...
        """
        super().__init__()
        self.units = units
        self.final_states = None
        self.init_state = None
        if isinstance(cell, str):
            if cell.upper() == 'GRU':
                cell = tf.keras.layers.GRUCell
            elif cell.upper() == 'LSTM':
                cell = tf.keras.layers.LSTMCell
            elif cell.upper() == 'RNN':
                cell = tf.keras.layers.SimpleRNNCell
            else:
                raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        self._cell = cell if hasattr(cell, 'units') else cell(units=units, dropout=timeseries_dropout)
        self.rnn = tf.keras.layers.RNN(cell=self._cell, *args, **kwargs)
        self.dropout = tf.keras.layers.Dropout(categorical_dropout)

        # multi cond (Serialization not implemented yet)
        max_num_conditions = 10
        self.multi_cond_to_init_state_dense = []
        for i in range(max_num_conditions):
            self.multi_cond_to_init_state_dense.append(tf.keras.layers.Dense(units=self.units, *args, **kwargs))
        self.multi_cond_p = tf.keras.layers.Dense(1, activation=None, use_bias=True, *args, **kwargs)

    def _standardize_condition(self, initial_cond):
        initial_cond_shape = initial_cond.shape
        if len(initial_cond_shape) == 2:
            initial_cond = tf.expand_dims(initial_cond, axis=0)
        first_cond_dim = initial_cond.shape[0]
        if isinstance(self._cell, tf.keras.layers.LSTMCell):
            if first_cond_dim == 1:
                initial_cond = tf.tile(initial_cond, [2, 1, 1])
            elif first_cond_dim != 2:
                raise Exception('Initial cond should have shape: [2, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        elif isinstance(self._cell, tf.keras.layers.GRUCell) or isinstance(self._cell, tf.keras.layers.SimpleRNNCell):
            if first_cond_dim != 1:
                raise Exception('Initial cond should have shape: [1, batch_size, hidden_size]\n'
                                'or [batch_size, hidden_size]. Shapes do not match.', initial_cond_shape)
        else:
            raise Exception('Only GRU, LSTM and RNN are supported as cells.')
        return initial_cond

    def build(self, input_shape):
        self.categorical_W = self.add_weight(name='categorical_kernel', shape=(input_shape[1][-1], self.units), initializer='glorot_uniform', trainable=True)
        self.categorical_b = self.add_weight(name='categorical_bias', shape=(self.units,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: List of n elements:
                    - [0] 3-D Tensor with shape [batch_size, time_steps, input_dim]. The inputs.
                    - [1:] list of tensors with shape [batch_size, cond_dim]. The conditions.
        In the case of a list, the tensors can have a different cond_dim.
        :return: outputs, states or outputs (if return_state=False)
        """
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        assert isinstance(inputs, list) and len(inputs) >= 2, f"{inputs}"
        x = inputs[0]
        cond = inputs[1:]
        if len(cond) > 1:  # multiple conditions.
            init_state_list = []
            for ii, c in enumerate(cond):
                init_state_list.append(self.multi_cond_to_init_state_dense[ii](self._standardize_condition(c)))
            multi_cond_state = self.multi_cond_p(tf.stack(init_state_list, axis=-1))
            multi_cond_state = tf.squeeze(multi_cond_state, axis=-1)
            self.init_state = tf.unstack(multi_cond_state, axis=0)
        else:
            cond = self._standardize_condition(cond[0])
            if cond is not None:
                # self.init_state = self.cond_to_init_state_dense_1(cond)
                self.init_state = tf.matmul(cond, self.categorical_W) + self.categorical_b
                self.init_state = tf.unstack(self.init_state, axis=0)
        for i in range(2):
            self.init_state[i] = self.dropout(self.init_state[i])
        out = self.rnn(x, initial_state=self.init_state, *args, **kwargs)
        if self.rnn.return_state:
            outputs, h, c = out
            final_states = tf.stack([h, c])
            return outputs, final_states
        else:
            return out

class SingleLayerConditionalRNN(tf.keras.Model):
    def __init__(self, NUM_CELLS, target_size, quantiles=_quantileList,
                categorical_dropout=0.0, timeseries_dropout=0.0, cell='LSTM',
                sigma=1., mu=0., ver='frozen', **kwargs):
        super().__init__()
        self.quantiles = quantiles
        self.target_size = target_size
        self.ver = ver
        self.layer1 = ConditionalRNN(NUM_CELLS, cell=cell,
                                    categorical_dropout=categorical_dropout,
                                    timeseries_dropout=timeseries_dropout,
                                    **kwargs)
        # self.layer2 = tf.keras.layers.Dropout(dropout)
        self.outs = [tf.keras.layers.Dense(self.target_size) for q in quantiles]
        self.conc = tf.keras.layers.Concatenate()
        self.rescale = Rescale_Transpose(sigma=sigma, mu=mu)
        if ver!='frozen': self.relu = tf.keras.layers.LeakyReLU(alpha=0.05)
        self.out = tf.keras.layers.Reshape((len(self.quantiles), self.target_size))

    def call(self, inputs, **kwargs):
        o = self.layer1(inputs)
        # o = self.layer2(o)
        o = [self.outs[_](o) for _ in range(len(self.quantiles))]
        o = self.conc(o)
        o = self.rescale(o)
        if self.ver!='frozen': o = self.relu(o)
        o = self.out(o)
        return o

class Rescale_Transpose(tf.keras.layers.Layer):
    def __init__(self, sigma=1., mu=0., name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.sigma = sigma
        self.mu = mu

    def call(self, inputs):
        dtype = self._compute_dtype
        sigma = tf.cast(self.sigma, dtype)
        mu = tf.cast(self.mu, dtype)
        return sigma * inputs + mu

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'sigma': self.sigma, 'mu': self.mu}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=0.2):
    """
    Return the splitting date (=number of training dates) for KFold splitting.
    Dates are assumed 1-step, 0-based.

    Parameters:
      history_size: int
        Size of history window.
      target_size: int
        Size of target window.
      total_size: int
        Total number of sample dates.
      split_ratio: int (default=0.2)
        K-fold ratio of train-validation split.

    Return:
      TRAIN_SPLIT: int
        # training dates (= index of splitting).
    """
    assert total_size>=2*history_size+target_size-1+(2/split_ratio), 'History and target sizes are too large.'

    return int((1-split_ratio)*(total_size-target_size+1)-(1-2*split_ratio)*history_size)+1

def train_val_split(data_ts, data_ctg, target_idx, history_size, target_size,
                    split_ratio=None, step_size=1, axis='time'):
    """
    Train-validation split.

    Parameters:
      data_ts: numpy ndarray
        Timeseries data of shape (# FIPS, length of timeline, # features)
      data_ctg: numpy ndarray
        Categorical data of shape (# FIPS, # features)
      target_idx: int
        Index of the target feature in data_ts.
      history_size: int
        Size of history window.
      target_size: int
        Size of target window.
      split_ratio: int (default=0.2)
        K-fold ratio of train-validation split.
      step_size: int (default=1)

    Return:
      X_train, C_train, y_train: numpy ndarray
        Timeseries, categorical, and target training data.
      X_val, C_val, y_val: numpy ndarray
        Timeseries, categorical, and target validation data.
    """
    total_size = len(data_ts[0])

    assert len(data_ts)==len(data_ctg), "Length of timeseries and categorical data do not match."

    if axis=='fips':
        if split_ratio is None:
            val_size = 1
        else:
            val_size = max(1, int(split_ratio * len(data_ts) // 1))
        val_set = random.sample(range(len(data_ts)), val_size)

        X_train, X_val = [], []
        y_train, y_val = [], []
        C_train, C_val = [], []

        for fips in range(len(data_ts)):
            if fips not in val_set:
                for i in range(history_size, total_size-target_size+1, step_size):
                    X_train.append(data_ts[fips][i-history_size:i, :])
                    C_train.append(data_ctg[fips])
                    y_train.append(data_ts[fips][i:i+target_size, target_idx])
            else:
                for i in range(history_size, total_size-target_size+1, step_size):
                    X_val.append(data_ts[fips][i-history_size:i, :])
                    C_val.append(data_ctg[fips])
                    y_val.append(data_ts[fips][i:i+target_size, target_idx])
    
    else:
        if split_ratio is None:
            TRAIN_SPLIT = total_size - history_size - target_size
        else:
            TRAIN_SPLIT = _get_TRAIN_SPLIT(history_size, target_size, total_size, split_ratio=split_ratio)
        print('TRAIN_SPLIT:', TRAIN_SPLIT)

        X_train, y_train = [], []
        X_val, y_val = [], []
        C_train, C_val = [], []

        for fips in range(len(data_ts)):
            for i in range(history_size, TRAIN_SPLIT, step_size):
                X_train.append(data_ts[fips][i-history_size:i, :])
                C_train.append(data_ctg[fips])
                y_train.append(data_ts[fips][i:i+target_size, target_idx])
            for i in range(TRAIN_SPLIT+history_size, total_size-target_size+1, step_size):
                X_val.append(data_ts[fips][i-history_size:i, :])
                C_val.append(data_ctg[fips])
                y_val.append(data_ts[fips][i:i+target_size, target_idx])

    return np.asarray(X_train), np.asarray(C_train), np.asarray(y_train), np.asarray(X_val), np.asarray(C_val), np.asarray(y_val)

def train_full(data_ts, data_ctg, target_idx, history_size, target_size, step_size=1):
    """
    Generate full training data for final model.

    Parameters:
      data_ts: numpy ndarray
        Timeseries data of shape (# FIPS, # features)
      data_ctg: numpy ndarray
        Categorical data of shape (# FIPS, # features)
      target_idx: int
        Index of the target feature in data_ts.
      history_size: int
        Size of history window.
      target_size: int
        Size of target window.
      step_size: int (default=1)

    Return:
      X_train, C_train, y_train: numpy ndarray
        Timeseries, categorical, and target training data.
    """
    total_size = len(data_ts[0])

    assert len(data_ts)==len(data_ctg), "Length of timeseries and categorical data do not match."

    X_train, C_train, y_train = [], [], []

    for fips in range(len(data_ts)):
        for i in range(history_size, total_size-target_size+1, step_size):
            X_train.append(data_ts[fips][i-history_size:i, :])
            C_train.append(data_ctg[fips])
            y_train.append(data_ts[fips][i:i+target_size, target_idx])

    return np.asarray(X_train), np.asarray(C_train), np.asarray(y_train)

def get_StandardScaler(X_train, X_ctg):
    """
    Instantiate and fit StandardScaler.

    Parameters:
      X_train, X_ctg: numpy ndarray
        Timeseries and categorical data of shape (# FIPS, length of timeline, # features).

    Return:
      scaler_ts, scaler_ctg: scikit-learn StandardScaler
        StandardScaler object fitted to X_train(resp. X_ctg).
    """
    scaler_ts, scaler_ctg = StandardScaler(), StandardScaler()
    scaler_ts.fit(np.vstack(X_train).astype(np.float32))
    scaler_ctg.fit(X_ctg.astype(np.float32))

    return scaler_ts, scaler_ctg

def normalizer(scaler, X):
    """
    Z-score the data.

    Parameters:
      scaler: scikit-learn StandardScaler
        Fitted StandardScaler object.
      X: numpy ndarray
        Either timeseries or categorical data.
    
    Return:
      Z-scored X
    """
    if len(X.shape)==2:
        X = scaler.transform(X)
    elif len(X.shape)==3:
        X = np.asarray(np.vsplit(scaler.transform(np.vstack(X)), len(X)))
    else:
        raise ValueError(f"Incompatible dimension of input ndarray. Input dimension: {len(X)}")
    return X

def load_Dataset(X_train, C_train, y_train, X_val=None, C_val=None, y_val=None, BATCH_SIZE=64, BUFFER_SIZE=100000, cache=False):
    """
    Generate training and validation datasets in the format of tensorflow Dataset class.

    Parameters:
      X_train, C_train, y_train: numpy ndarray
        Timeseries, categorical, and target training data.
      X_val, C_val, y_val: numpy ndarray (default=None)
        Timeseries, categorical, and target validation data.
      BATCH_SIZE: int (default=64)
        Size of each batch.
        Popular BATCH_SIZE: 32, 64, 128
        Oftentimes smaller BATCH_SIZE performs better.
      BUFFER_SIZE: int (default=100000)
        Size of buffer. Determines the quality of permutation.
      cache: bool (default=True)
        True if cache training data on the memory. Turn it False if there is memory issue.

    Return:
      train_data, val_data: tensorflow Dataset
    """
    X_tr_data = tf.data.Dataset.from_tensor_slices(X_train)
    C_tr_data = tf.data.Dataset.from_tensor_slices(C_train)
    y_tr_data = tf.data.Dataset.from_tensor_slices(y_train)
    train_data = tf.data.Dataset.zip(((X_tr_data, C_tr_data), y_tr_data))
    if cache:
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    else:
        train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    if X_val is None:
        return train_data
    else:
        X_v_data = tf.data.Dataset.from_tensor_slices(X_val)
        C_v_data = tf.data.Dataset.from_tensor_slices(C_val)
        y_v_data = tf.data.Dataset.from_tensor_slices(y_val)
        val_data = tf.data.Dataset.zip(((X_v_data, C_v_data), y_v_data))
        val_data = val_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        
        return train_data, val_data

def quantileLoss(quantile, y, y_p):
    """
    Custum loss function for quantile forecast models.
    Intended usage:
    >>> loss=lambda y_p, y: quantileLoss(quantile, y_p, y)
    in compile step.

    Parameters:
      quantile: float in [0,1]
        Quantile number.
    """
    e = y - y_p
    return tf.math.reduce_mean(tf.math.maximum(quantile*e, (quantile-1)*e))

def MultiQuantileLoss(quantiles, target_size, y, y_p):
    """
    quantileLoss adapted for multi-output conditional RNN.

    Parameters:
      quantiles: list of floats in [0,1]
        List of quantiles.
      target_size: int
        Size of target window.
    """
    return tf.math.reduce_mean(tf.stack([quantileLoss(quantiles[_], y, y_p[:, _, :]) for _ in range(len(quantiles))]))

def LSTM_fit(train_data, val_data=None, lr=0.001, NUM_CELLS=128, EPOCHS=10, dp_ctg=0.2, dp_ts=0.2, monitor=False, callbacks=None, **kwargs):
    """
    Build and fit the conditional LSTM model.

    Parameters:
      train_data: tensorflow Dataset
        Training dataset.
      val_data: tensorflow Dataset (default=None)
        Validation dataset, optional.
      lr: float (default=0.001)
        Learning rate.
      NUM_CELLS: int (default=128)
        Number of cells in LSTM layer.
      EPOCHS: int (default=10)
        Number of epochs for training.
      df: float in [0,1] (default=0.2)
        Dropout rate.
      monitor: bool (default=False)
        True if return the history of model fitting.
      callbacks: list (default=None)
        List of tensorflow callbacks classes.
      **kwargs:
        Additional kwargs are passed to tensorflow Model.

    Return:
      model_qntl: list of tensorflow Model
        List of fitted conditional LSTM models.
      history_qntl: list of tensorflow History
        List of the history of model fitting.
    """
    target_size = train_data.element_spec[1].shape[1]

    celltype = kwargs["cell"] if "cell" in kwargs else "LSTM"
    mu = kwargs["mu"] if "mu" in kwargs else 0
    sigma = kwargs["sigma"] if "sigma" in kwargs else 1
    for key in ["cell", "mu", "sigma"]:
        try:
            del kwargs[key]
        except KeyError:
            continue

    model_qntl = [SingleLayerConditionalRNN(NUM_CELLS, target_size,
                                            categorical_dropout=dp_ctg,
                                            timeseries_dropout=dp_ts,
                                            cell=celltype, sigma=sigma, mu=mu)
                                            for _ in range(len(_quantileList))]
    history_qntl =[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(len(_quantileList)):
        model_qntl[i].compile(optimizer=optimizer, loss=lambda y, y_p: quantileLoss(_quantileList[i], y, y_p))
        print(f'Quantile={10*(i+1)} is trained')

    for i in range(len(_quantileList)):
        if val_data is None:
            history = model_qntl[i].fit(train_data, epochs=EPOCHS, steps_per_epoch=2500, callbacks=callbacks, **kwargs)
        else:
            history = model_qntl[i].fit(train_data, epochs=EPOCHS, steps_per_epoch=2500, validation_data=val_data,
                                        validation_steps=200, callbacks=callbacks, **kwargs)
        history_qntl.append(history)

    if monitor:
        return model_qntl, history_qntl
    else:
        return model_qntl

def LSTM_fit_mult(train_data, val_data=None, hparam=None, monitor=False, callbacks=None, ver='frozen', **kwargs):
    """
    Build and fit the multi-output conditional LSTM model.

    Parameters:
      train_data: tensorflow Dataset
        Training dataset.
      val_data: tensorflow Dataset (default=None)
        Validation dataset, optional.
      lr: float (default=0.001)
        Learning rate.
      NUM_CELLS: int (default=128)
        Number of cells in LSTM layer.
      EPOCHS: int (default=10)
        Number of epochs for training.
      df: float in [0,1] (default=0.2)
        Dropout rate.
      monitor: bool (default=False)
        True if return the history of model fitting.
      callbacks: list (default=None)
        List of tensorflow callbacks classes.
      **kwargs:
        Additional kwargs are passed to tensorflow Model.

    Return:
      model: tensorflow Model
        Fitted multi-output conditional LSTM models.
      history: tensorflow History
        The history of model fitting.
    """
    target_size = train_data.element_spec[1].shape[1]

    celltype = kwargs["cell"] if "cell" in kwargs else "LSTM"
    mu = kwargs["mu"] if "mu" in kwargs else 0
    sigma = kwargs["sigma"] if "sigma" in kwargs else 1
    for key in ["cell", "mu", "sigma"]:
        try:
            del kwargs[key]
        except KeyError:
            continue

    hparam_default = {
        "lr": 0.001,
        "NUM_CELLS": 128,
        "EPOCHS": 10,
        "dp_ctg": 0.2,
        "dp_ts": 0.2
    }
    if hparam is None: hparam = hparam_default
    for key in hparam_default:
        if key not in hparam: hparam[key] = hparam_default[key]

    model = SingleLayerConditionalRNN(hparam["NUM_CELLS"], target_size,
                                    categorical_dropout=hparam["dp_ctg"],
                                    timeseries_dropout=hparam["dp_ts"],
                                    cell=celltype, sigma=sigma, mu=mu, ver=ver)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparam["lr"])

    model.compile(optimizer=optimizer, loss=lambda y, y_p: MultiQuantileLoss(_quantileList, target_size, y, y_p))
    logger.info('Training multi-output model.')

    if val_data is None:
        history = model.fit(train_data, epochs=hparam["EPOCHS"], callbacks=callbacks, **kwargs)
    else:
        history = model.fit(train_data, epochs=hparam["EPOCHS"], validation_data=val_data, callbacks=callbacks, **kwargs)
    logger.info('Training complete.')

    if monitor:
        return model, history
    else:
        return model

def LSTM_finder(train_data, val_data, lr=0.001, NUM_CELLS=128, EPOCHS=10, dp=0.2):
    target_size = train_data.element_spec[1].shape[1]
    
    model_qntl = [SingleLayerConditionalRNN(NUM_CELLS, target_size, dropout=dp) for _ in range(len(_quantileList))]
    history_qntl =[]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    lr_finder = [LRFinder() for _ in range(len(_quantileList))]

    for i in range(len(_quantileList)):
        model_qntl[i].compile(optimizer=optimizer, loss=lambda y_p, y: quantileLoss(_quantileList[i], y_p, y))
        print(f'Quantile={10*(i+1)} is trained')
        history = model_qntl[i].fit(train_data, epochs=EPOCHS, validation_data=val_data, validation_steps=300, callbacks=[lr_finder[i]])

    return lr_finder

def predict_future(model_qntl, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=None, date_ed=None):
    """
    Forecast the future using the fitted conditional LSTM model.

    Parameters:
      model_qntl: list of tensorflow Model
        Fitted conditional LSTM models.
      data_ts, data_ctg: numpy ndarray
        Timeseries and categorical data with respect to the last history window.
      scaler_ts, scaler_ctg: scikit-learn StandardScaler
        Fitted StandardScaler. Needed to undo the Z-scoring.
      history_size: int
        Size of history window.
      target_idx: int
        Index of the target feature in data_ts.
      FIPS: list (default=None)
        List of county FIPS, sorted to match with the datasets.
      date_ed: pandas Timestamp
        Last date in data_ts(= one day before the start date of forecast).

    Return:
        pandas Dataframe of the prediction in tidy format.
    """
    X_future = [data[-history_size:, :] for data in data_ts]; X_future = np.asarray(X_future)
    C_future = data_ctg; C_future = np.asarray(C_future)

    X_future = np.asarray(np.vsplit(scaler_ts.transform(np.vstack(X_future)), len(X_future)))
    C_future = scaler_ctg.transform(C_future)

    prediction_future = []
    for i in range(len(model_qntl)):
        prediction_future.append(model_qntl[i].predict((X_future, C_future)))
    prediction_future = np.asarray(prediction_future)
    target_size = prediction_future.shape[2]

    if (FIPS is None) or (date_ed is None):
        return np.asarray(prediction_future)
    else:
        print('Saving future prediction.')
        df_future = []
        for i, fips in enumerate(FIPS):
            for j in range(target_size):
                df_future.append([date_ed+pd.Timedelta(days=1+j), fips]+prediction_future[:,i,j].tolist())

        return pd.DataFrame(df_future, columns=['date', 'fips']+list(range(10, 100, 10)))

def predict_future_mult(model, data_ts, data_ctg, scaler_ts, scaler_ctg, history_size, target_idx, FIPS=None, date_ed=None):
    """
    Forecast the future using the fitted multi-output conditional LSTM model.

    Parameters:
      model: tensorflow Model
        Fitted multi-output conditional LSTM models.
      data_ts, data_ctg: numpy ndarray
        Timeseries and categorical data with respect to the last history window.
      scaler_ts, scaler_ctg: scikit-learn StandardScaler
        Fitted StandardScaler. Needed to undo the Z-scoring.
      history_size: int
        Size of history window.
      target_idx: int
        Index of the target feature in data_ts.
      FIPS: list (default=None)
        List of county FIPS, sorted to match with the datasets.
      date_ed: pandas Timestamp
        Last date in data_ts(= one day before the start date of forecast).

    Return:
        pandas Dataframe of the prediction in tidy format.
    """
    X_future = [data[-history_size:, :] for data in data_ts]; X_future = np.asarray(X_future)
    C_future = data_ctg; C_future = np.asarray(C_future)

    X_future = np.asarray(np.vsplit(scaler_ts.transform(np.vstack(X_future)), len(X_future)))
    C_future = scaler_ctg.transform(C_future)

    prediction_future = model.predict((X_future, C_future))

    if (FIPS is None) or (date_ed is None):
        return prediction_future
    else:
        logger.info('Saving future prediction.')
        df_future = []
        for i, fips in enumerate(FIPS):
            for j in range(prediction_future.shape[2]):
                df_future.append([date_ed+pd.Timedelta(days=1+j), fips]+prediction_future[i,:,j].tolist())

        return pd.DataFrame(df_future, columns=['date', 'fips']+[str(10*i) for i in range(1,10)])

def plot_train_history(history, title='Untitled', path=None):
    """
    Plot the learning curve.

    Parameters:
      history: tensorflow History
        History of model fitting.
      title: str (default='Untitled')
        Title of the figure.
      path: str (default=None)
        Path to save the learning curve. Set None to show instead of save the figure.
    """
    loss = history.history['loss']
    is_val_loss = True
    try:
        val_loss = history.history['val_loss']
    except:
        is_val_loss = False

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    if is_val_loss:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

def plot_prediction(model_qntl, val_data, scaler, target_idx, num=3):
    mu, sigma = scaler.mean_[target_idx], scaler.scale_[target_idx]
    
    for x, y in val_data.take(num):
        X_unnm = scaler.inverse_transform(x)
        y_unnm = sigma * y + mu
        
        plt.figure(figsize=(12,6))
        plt.plot(list(range(-len(X_unnm[0]), 0)), np.array(X_unnm[0][:, target_idx]), label='History')
        plt.plot(np.arange(len(y_unnm[0])), np.array(y_unnm[0]), 'bo', label='True Future')
        for i in [0, 4, 8]:
            prediction = model_qntl[i].predict(X_unnm)[0]
            prediction = sigma * prediction + mu
            plt.plot(np.arange(len(y_unnm[0])), np.array(prediction), 'o', label=f'Predicted Future, qntl={10*(i+1)}')
        plt.legend(loc='upper left')
        plt.show()