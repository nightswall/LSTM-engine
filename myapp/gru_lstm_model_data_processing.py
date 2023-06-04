import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn.preprocessing import MinMaxScaler


#datasetFileName = "myapp/Occupancy_source.csv"

def getTrainLoaderFirstTime(filename,attribute_number):
    #use enum like (dictionary) to attribute number 
    #pd.read_csv(filename).head()
    # The scaler objects will be stored in this dictionary so that our output test data from the model can be re-scaled during evaluation

    train_x = []
    test_x = {}
    test_y = {}

    df=pd.read_csv(filename,parse_dates=['DateTime'])
    #print(df.index)
    # Processing the time data into suitable input formats,
    # Processing the time data into suitable input formats,
    #df['DateTime'] = pd.to_datetime(df['DateTime'])
    # add a 'totalseconds' column using the 'total_seconds' method
    #df['totalseconds'] = df['DateTime'].apply(lambda x: x.timestamp())
    df['minute'] = df['DateTime'].apply(lambda x: '{:02d}'.format(x.minute)).astype(int)
    df['hour'] = df.apply(lambda x: x['DateTime'].hour,axis=1)
    df['dayofweek'] = df.apply(lambda x: x['DateTime'].dayofweek,axis=1)
    df['month'] = df.apply(lambda x: x['DateTime'].month,axis=1)
    #df['dayofyear'] = df.apply(lambda x: x['DateTime'].dayofyear,axis=1)
    df = df.sort_values('DateTime').drop('DateTime',axis=1)
    df['Bus'] = df['Bus'].map({'Bus 0': 0, 'Bus 1': 1, 'Bus 2': 2, 'Bus 3': 3, 'Bus 4': 4, 'Bus 5': 5  }) ## For Bus mapping
    #df = df.drop('Voltage',axis=1)
    #df = df.drop('Temperature',axis=1)
    #df = df.drop('Current',axis = 1)


    # Scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)

    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(df.iloc[:,attribute_number].values.reshape(-1,1))
    lookback= 12
    inputs = np.zeros((len(data)-lookback,lookback,df.shape[1]))
    labels = np.zeros(len(data)-lookback)
    for i in range(lookback, len(data)):
        inputs[i-lookback] = data[i-lookback:i]
        labels[i-lookback] = data[i,attribute_number]

    inputs = inputs.reshape(-1,lookback,df.shape[1])
    labels = labels.reshape(-1,1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.1*len(inputs))
    #if len(train_x) == 0:
    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]
    #else:
    #    train_x = np.concatenate((train_x,inputs[:-test_portion]))
    #    train_y = np.concatenate((train_y,labels[:-test_portion]))

    batch_size = 2
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    #print( train_loader, sc, label_sc ,df)

    return train_loader, sc, label_sc, df


def getTrainLoaderLater(filename, datasetFileName, attribute_number):
    # Get Original Dataset pop first 100 records push new 100 records from temp csv
    new_temp_df = pd.read_csv(filename,parse_dates=[0])

    source_df = pd.read_csv(datasetFileName,parse_dates=[0])

    #source_df
    source_df = source_df.drop(source_df.index[:50])
    #print("\nDataFrame after dropping first 100 records:\n", source_df.head())

    # add new 100 records to end from new DataFrame
    source_df = pd.concat([source_df, new_temp_df.head(50)]) #.head(100)?
    #print("\nDataFrame after adding new 100 records:\n", source_df.tail())

    #Overwrite into csv file
    with open(datasetFileName, 'w') as f:
        source_df.to_csv(f, index=False)

    #use enum like (dictionary) to attribute number 
    #pd.read_csv(filename).head()

    # The scaler objects will be stored in this dictionary so that our output test data from the model can be re-scaled during evaluation

    train_x = []
    test_x = {}
    test_y = {}

    df=pd.read_csv(datasetFileName,parse_dates=['DateTime'])
    #print(df.index)
    # Processing the time data into suitable input formats,
    # Processing the time data into suitable input formats,
    #df['DateTime'] = pd.to_datetime(df['DateTime'])
    # add a 'totalseconds' column using the 'total_seconds' method
    #df['totalseconds'] = df['DateTime'].apply(lambda x: x.timestamp())
    df['minute'] =  df['DateTime'].apply(lambda x: '{:02d}'.format(x.minute)).astype(int)
    df['hour'] = df.apply(lambda x: x['DateTime'].hour,axis=1)
    df['dayofweek'] = df.apply(lambda x: x['DateTime'].dayofweek,axis=1)
    df['month'] = df.apply(lambda x: x['DateTime'].month,axis=1)
    #df['dayofyear'] = df.apply(lambda x: x['DateTime'].dayofyear,axis=1)
    df = df.sort_values('DateTime').drop('DateTime',axis=1)
    df['Bus'] = df['Bus'].map({'Bus 0': 0, 'Bus 1': 1, 'Bus 2': 2, 'Bus 3': 3, 'Bus 4': 4, 'Bus 5': 5  }) ## For Bus mapping

    # Scaling the input data
    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(df.values)
   
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(df.iloc[:,attribute_number].values.reshape(-1,1))
    lookback=12
    inputs = np.zeros((len(data)-lookback,lookback,df.shape[1]))
    labels = np.zeros(len(data)-lookback)
    for i in range(lookback, len(data)):
        inputs[i-lookback] = data[i-lookback:i]
        labels[i-lookback] = data[i,attribute_number]

    inputs = inputs.reshape(-1,lookback,df.shape[1])
    labels = labels.reshape(-1,1)

    # Split data into train/test portions and combining all data from different files into a single array
    test_portion = int(0.1*len(inputs))
    #if len(train_x) == 0:
    train_x = inputs[:-test_portion]
    train_y = labels[:-test_portion]
    #else:
    #    train_x = np.concatenate((train_x,inputs[:-test_portion]))
    #    train_y = np.concatenate((train_y,labels[:-test_portion]))

    batch_size = 2
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    #print( train_loader, sc, label_sc ,df)

    return train_loader, sc, label_sc, df


