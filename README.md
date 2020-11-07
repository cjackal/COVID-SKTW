# COVID-19 Prediction Model condLSTM-Q

## Overview

A LSTM-based prediction model for daily COVID-19 death counts.

## Usage

Set `config.json` accordingly.

1. `date_generated`: Dummy in effect.

2. `start_train`: The first date of the input data to be trained. `null` uses the earliest possible date.

3. `end_train`: The last date of the input data to be trained.

4. `start_date`: The first date of prediction.

5. `end_date`: The last date of prediction. `null` defaults to 2-week prediction.

6. `hparam`: One can pass custom hyperparameters to the model in the form of `parameter name`:`value` object.

    Currently supported:

    `history_size`: size of the history window

    `NUM_CELLS`: number of cells of LSTM layer

    `lr`: learning rate

    `dp_ctg`: dropout rate on categorical inputs

    `dp_ts`: dropout rate on timeseries inputs

    `EPOCHS`: training epochs

7. `out_files`: Path to output forecast (in `csv` format).

Run `main.py`.

> One can pass command line arguments.

```cmd
python main.py <version name, optional> <path/to/config.json, optional>
```
