# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:49:10 2020

@author: HyeongChan Jo
"""
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import tensorflow as tf
from cond_rnn import ConditionalRNN



def concatDF_static(data1, column1, fipsList, df_death_accum, minDeathNumber = 0, dic_cluster = []):
    # inputs
    #   data:               list of input dataframes
    #   columnName:         list of lists storing column names of input dataframes to be used
    #   fipsList:           list of fips of the counties to be used
    #   minDeathNumber:     minimum number of accumulative death on the last day that the dataset should satisfy - counties with less number of death will be removed
    
    # divide the fipslist based on the minDeathNumber
    totalNumDeath = [df_death_accum[x].iloc[-1, -1] for x in range(len(df_death_accum))]
    fipsList = [fipsList[i] for i in range(len(fipsList)) if totalNumDeath[i]>=minDeathNumber]
    
    # save the original data
    data1_orig = data1.copy()
    
    # go over each FIPS value
    dataList = []
    fips_noData = []
    fips_final = []
    for i, fips in enumerate(fipsList):
        # for data1 (demographic)
        try: 
            data1 = data1_orig.loc[fips, column1].to_numpy()
            if len(dic_cluster)!=0:
                data1 = np.append(data1, dic_cluster[fips])
        except  KeyError:
            print('fips ', fips, 'not found')
            fips_noData.append(fips)
            continue
        fips_final.append(fips)
        
        dataList.append(data1)
        
    return dataList, fips_noData, fips_final





def concatDF_timeseries(data2, data3, data4, column2, column3, column4, fipsList, df_fips2state, data_demo, fipsList_demo, date_st, date_ed, 
             df_death_accum, minDeathNumber = 0, smoothData = False, exp = 2, removeNeg = True, normalizeTarget = False):
    # inputs
    #   data:               list of input dataframes
    #   columnName:         list of lists storing column names of input dataframes to be used
    #   fipsList:           list of fips of the counties to be used
    #   fipsList_demo:      list of fips of the counties that were found in demogrpahic dataset
    #   date_st:            starting date of the dataset to be used, in datetime variable
    #   date_ed:            ending date of the dataset to be used, in datetime variable
    #   minDeathNumber:     minimum number of accumulative death on the last day that the dataset should satisfy - counties with less number of death will be removed
    #   smoothData:         whether to smooth the data with exponential filter
    #   exp:                base of the exponential filter
    #   removeNeg:          if it's true, change negative values into zeros when deaths<0
    #   normalizeTarget:    if it's true, normalize the target (deaths) too
    
    # divide the fipslist based on the minDeathNumber
    totalNumDeath = [df_death_accum[x].iloc[-1, -1] for x in range(len(df_death_accum))]
    fipsList = [fipsList[i] for i in range(len(fipsList)) if totalNumDeath[i]>=minDeathNumber]
    data2 = [data2[i] for i in range(len(fipsList)) if fipsList[i] in fipsList]

    # handle dates
    date_st_str = date_st.strftime("%Y-%m-%d")
    date_ed_str = date_ed.strftime("%Y-%m-%d")
    numDays = (date_ed-date_st).days+1
    date_st_seasonality = date_st - relativedelta(years=3)  # seasonality data has been saved based on 2017 data
    date_st_seasonality = date_st_seasonality.strftime("%Y-%m-%d")
    date_ed_seasonality = date_ed - relativedelta(years=3)
    date_ed_seasonality = date_ed_seasonality.strftime("%Y-%m-%d")
    
    # save the original data
    data2_orig = data2.copy()
    data3_orig = data3.copy()
    data4_orig = data4.copy()
    
    # go over each FIPS value
    dataList = []
    fips_noData = []
    fips_final = []
    for i, fips in enumerate(fipsList):
        if fips not in fipsList_demo:
            continue
        
        # for data2 (mortality)
        data2 = data2_orig[i].loc[date_st_str:date_ed_str, column2].to_numpy()
        if data2.shape[0]<numDays:
            data2 = np.concatenate( (np.zeros((numDays-data2.shape[0], len(column2))), data2), axis = 0)        
        if removeNeg: 
            data2[data2[:,1]<0, 1] = 0
        if smoothData:
            for i in range(data2.shape[0]):
                if i == 0: continue
                data2[i, 1] = data2[i, 1]+data2[i-1, 1]/exp
        
        # for data3 (mobility)
        if len(data3_orig[i]) == 0:
            # temporary solution, assuming that they are located close to each other
            d = 1
            while True:
                if len(data3_orig[i-d])==0:
                    d+=1
                else:
                    data3 = data3_orig[i-d].loc[date_st_str:date_ed_str, column3].to_numpy()
                    break
        else:
            data3 = data3_orig[i].loc[date_st_str:date_ed_str, column3].to_numpy()
        if len(data3)<len(data2):
            data3 = np.vstack( (data3, np.kron( np.ones((len(data2)-len(data3),1)), data3[-1, :]) ) )
        
        # for data4 (seasonality)
        state = df_fips2state.loc[fips, 'state']
        data4 = data4_orig.loc[state, :]
        data4.set_index('date', inplace=True)
        data4 = data4.loc[date_st_seasonality:date_ed_seasonality, column4]
        
        data = np.hstack( (data2, data3, data4) )
        
        dataList.append(data)
        
    return dataList, fips_noData, fips_final




def dataPrep_LSTM_fromNP_static(dataList, dataList_time, input_size, output_size, step_size, testingSet_size):
    ## inputs
    ##  - dataList: list of dataframes for training/testing the model
    ##  - dataList_time: list of dataframes with timeseries data. Used to set up a correct batch size with this function
    ##  - targetName: name of the column to be used for prediction
    ##  - input_size: size of the input data
    ##  - output_size: size of the output data (i.e. horizon)
    ##  - step_size: step size between each of training dataset
    ##  - testingSet_size: size of the validation set, in terms of the number of outputs
    ##  - columnName: name of each column
    
    # initialization
    trainingData = []
    testingData = []
    
    for dfIdx in range(len(dataList)):
        df = dataList[dfIdx]
        df_time = dataList_time[dfIdx]
        
        for i in range(input_size, len(df_time)-testingSet_size-output_size, step_size):
            trainingData.append( df )
        
        for i in range(len(df)-testingSet_size, len(df)-output_size+1, step_size):
            testingData.append( df )
            
    return np.array(trainingData).astype(np.float32), np.array(testingData).astype(np.float32)




def dataPrep_LSTM_fromNP_time(dataList, targetName, input_size, output_size, step_size, testingSet_size, columnName):
    ## inputs
    ##  - dataList: list of dataframes for training/testing the model
    ##  - targetName: name of the column to be used for prediction
    ##  - input_size: size of the input data
    ##  - output_size: size of the output data (i.e. horizon)
    ##  - step_size: step size between each of training dataset
    ##  - testingSet_size: size of the validation set, in terms of the number of outputs
    ##  - columnName: name of each column
    
    # initialization
    trainingData = []
    trainingAns = []
    testingData = []
    testingAns = []
    
    # find the target column
    targetIdx = columnName.index(targetName)
    
    for df in dataList:
        for i in range(input_size, len(df)-testingSet_size-output_size, step_size):
            trainingData.append( df[i-input_size : i, :] )
            trainingAns.append( df[i : i+output_size, targetIdx] )
        
        for i in range(len(df)-testingSet_size, len(df)-output_size+1, step_size):
            testingData.append( df[i-input_size : i, :] )
            testingAns.append( df[i : i+output_size, targetIdx] )
            
    return np.array(trainingData).astype(np.float32), np.array(trainingAns).astype(np.float32), np.array(testingData).astype(np.float32), np.array(testingAns).astype(np.float32)




class condLSTM(tf.keras.Model):
    def __init__(self, NUM_CELLS, NUM_DAYS):
        super(condLSTM, self).__init__()
        self.cond = ConditionalRNN(NUM_CELLS, cell='LSTM', dtype=tf.float32)
        self.out = tf.keras.layers.Dense(units=NUM_DAYS)

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = self.out(o)
        return o

def quantileLoss(quantile, y_p, y):
    e = y_p - y
    return tf.keras.backend.mean(tf.keras.backend.maximum(quantile*e, (quantile-1)*e))


class condLSTM_quantile(tf.keras.Model):
    def __init__(self, NUM_CELLS, NUM_DAYS, quantileList):
        super(condLSTM_quantile, self).__init__()
        self.cond = ConditionalRNN(NUM_CELLS, cell='LSTM', dtype=tf.float32)
        self.out = tf.keras.layers.Dense(units=NUM_DAYS)
        self.quantileList = quantileList
        self.num_quantiles = len(quantileList)
        self.outputs = []
        self.losses = []
        
        # Create outputs and losses for all quantiles
        for i in range(self.num_quantiles):
            q = self.quantileList[i]
            
            # Get output layers 
            output = tf.keras.layers.Dense(units=NUM_DAYS)
            self.outputs.append(output)
            
            # Create losses
            error = tf.subtract(self.y, output)
            loss = tf.reduce_mean(tf.maximum(q*error, (q-1)*error), axis=-1)

            self.losses.append(loss)

        # Create combined loss
        self.combined_loss = tf.reduce_mean(tf.add_n(self.losses))
        self.train_step = tf.train.AdamOptimizer().minimize(self.combined_loss)

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = self.out(o)
        return o
    
    
    
    
    
    
class condLSTM_quantile_test(tf.keras.Model):
    def __init__(self, NUM_CELLS, NUM_DAYS, quantileList, activation = 'relu'):
        super(condLSTM_quantile_test, self).__init__()
        self.cond = ConditionalRNN(NUM_CELLS, cell='LSTM', activation=activation, dtype=tf.float32)
        self.outs = []
        self.losses = []
        self.quantileList = quantileList
        self.num_quantiles = len(quantileList)
        
        # Create outputs and losses for all quantiles
        for i in range(self.num_quantiles):
            q = self.quantileList[i]
            
            # Get output layers 
            out = tf.keras.layers.Dense(units=NUM_DAYS)
            self.outs.append(out)
            
            # Create losses
            error = tf.subtract(self.y, output)
            loss = tf.reduce_mean(tf.maximum(q*error, (q-1)*error), axis=-1)

            self.losses.append(loss)

        # Create combined loss
        self.combined_loss = tf.reduce_mean(tf.add_n(self.losses))
        self.train_step = tf.train.AdamOptimizer().minimize(self.combined_loss)

    def call(self, inputs, **kwargs):
        o = self.cond(inputs)
        o = [x(o) for x in self.outs]
        return o
    
def pinballLoss(y_true, y_pred):
    error = [y_true - y for y in y_pred]