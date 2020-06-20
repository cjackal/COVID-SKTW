## utility.py

Module containing auxillary functions to be used in other modules. This module is imported on most other modules.

## DataCleaner.py

Designed to run on daily basis. It reads up the datasets to be trained, and generate the preprocessed data to be used in LSTM under the folder preprocessing/{TODAY}. During the run, it prints the starting and end date to be included in the training, which should be fed back to LSTM training.

Fully capitalized variables at the beginning of the codes (that is, the paths to the datasets) are the only parts that can vary.

## LSTM.py

Module containing all the definition of necessary functions and classes for LSTM.

Currently available: All except possibly plot_prediction function.
Naming conventions should be clear, but just in case, a few elaborations follow:

class LRFinder: used in LSTM_finder function. See the comment in the code for details.

class conditionalRNN: a keras Layer. Used in all LSTM model build-up steps. See the comment in the code for details.

class SingleLayerConditionalRNN: a keras model consisting of single conditional RNN followed by the output dense layer.

function load_Dataset: it generates a tensorflow Dataset class from preprocessed datasets (which is in the form of numpy array).

function LSTM_fit: it trains the conditional RNN quantile-wise, the output is of the form of a list of tensorflow Models (of length 9).

function LSTM_finder: it runs a quick training of conditional RNN by varying the learning rate and plots the validation error. The learning rate hitting the minimum is supposed to be the optimal learning rate.

function predict_future: predict the future using the trained model (from LSTM_fit, say). Output is a pandas dataframe, which can be transformed to the submission format using to_multi_idx function in utility.py

## LSTM_trainer.py

The front-end module that do the actual training. Don't forget to change the hyperparameters into what you want.
Especially, don't forget to set TODAY and date_ed variables.
