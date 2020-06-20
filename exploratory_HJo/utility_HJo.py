# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:49:10 2020

@author: HyeongChan Jo
"""
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime

def dataPrep_LSTM_fromDF(dataList, targetName, input_size, output_size, step_size, testingSet_size):
    ## inputs
    ##  - dataList: list of dataframes for training/testing the model
    ##  - targetName: name of the column to be used for prediction
    ##  - input_size: size of the input data
    ##  - output_size: size of the output data (i.e. horizon)
    ##  - step_size: step size between each of training dataset
    ##  - testingSet_size: size of the validation set, in terms of the number of outputs
    
    # initialization
    trainingData = []
    trainingAns = []
    testingData = []
    testingAns = []
    
    for df in dataList:
        for i in range(input_size, len(df)-testingSet_size-output_size, step_size):
            trainingData.append(np.array(df[i-input_size : i, :]))
            trainingAns.append(np.array(df[i : i+output_size, targetName]))
        
        for i in range(len(df)-testingSet_size, len(df)-output_size, step_size):
            testingData.append(np.array(df[i-input_size : i, :]))
            testingAns.append(np.array(df[i : i+output_size, targetName]))
            
    return np.array(trainingData), np.array(trainingAns), np.array(testingData), np.array(testingAns)


def dataPrep_LSTM_fromNP(dataList, targetName, input_size, output_size, step_size, testingSet_size, columnName):
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



def concatDF_test(data1, data2, data3, data4, column1, column2, column3, column4, fipsList, date_st, date_ed, 
             df_death_accum, minDeathNumber = 0, smoothData = False, exp = 2, removeNeg = True, normalizeTarget = False):
    # inputs
    #   data:               list of input dataframes
    #   columnName:         list of lists storing column names of input dataframes to be used
    #   fipsList:           list of fips of the counties to be used
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
    data1_orig = data1.copy()
    data2_orig = data2.copy()
    data3_orig = data3.copy()
    data4_orig = data4.copy()
    
    # go over each FIPS value
    dataList = []
    fips_noData = []
    fips_final = []
    for i, fips in enumerate(fipsList):
        # for data1 (demographic)
        try: 
            data1 = data1_orig.loc[fips, column1].to_numpy()
            data1 = np.kron( np.ones( (numDays, 1) ), data1)
        except  KeyError:
            print('fips ', fips, 'not found')
            fips_noData.append(fips)
            continue
        fips_final.append(fips)
        
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
        state = data1_orig.loc[fips, 'State']
        state = state[0:-1]
        data4 = data4_orig.loc[state, :]
        data4.set_index('date', inplace=True)
        data4 = data4.loc[date_st_seasonality:date_ed_seasonality, column4]
        
        data = np.hstack( (data1, data2, data3, data4) )
        
        dataList.append(data)
        
    return dataList, fips_noData, fips_final


# normalization
def normalizeData(dataList):
    data_all = np.vstack((dataList)).astype(np.float32)

    data_all_mean = data_all.mean(axis = 0)
    data_all_std = data_all.std(axis = 0)

    dataList_zscore = [(data-data_all_mean)/data_all_std for data in dataList]
    
    return dataList_zscore, data_all_mean, data_all_std


# if normalization is not required for target values
def unnormalizeTarget(trainingAns, testingAns, data_all_mean, data_all_std, columnName, target):
    targetIdx = columnName.index(target)
    
    trainingAns = trainingAns*data_all_std[targetIdx] + data_all_mean[targetIdx]
    testingAns = testingAns*data_all_std[targetIdx] + data_all_mean[targetIdx]
    
    return trainingAns, testingAns