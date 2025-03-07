import pandas as pd 
import numpy as np 

import sys
import os

path= "C:\\Users\\Aman\\Desktop\\MODIFIED-FRAUD-DETECTION\\src"
sys.path.append(os.path.abspath(path=path))

try:
    from data_preprocessing import DataPreprocessing
except:
    print("Import failure")


class featureEngineering:
    def __init__(self, data):
        self.data = data
    def feature_extraction(self):
        #  Sort and compute time differences
        self.data = self.data.sort_values(by=['device_id', 'purchase_time'])
        self.data['time_diff(hr)'] = self.data.groupby('device_id')['purchase_time'].diff().dt.total_seconds()/3600
        self.data['time_diff(hr)'] =self.data['time_diff(hr)'].round(2)

        # Compute average transaction velocity
        avg_velocity = self.data.groupby('device_id')['time_diff(hr)'].mean().reset_index(name='avg_transaction_velocity')
        avg_velocity['avg_transaction_velocity(Hr)'] = avg_velocity['avg_transaction_velocity']

        avg_velocity.fillna(0, inplace=True)
        avg_velocity.drop('avg_transaction_velocity', axis=1)

        self.data['hour_of_day'] = self.data['purchase_time'].dt.hour
        self.data['day_of_week'] = self.data['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6
        # Frequency Encoding
        device_id_freq = self.data['device_id'].value_counts(normalize=False)
        self.data['device_id_encoded'] = self.data['device_id'].map(device_id_freq)
        self.data['country'].fillna('undefined', inplace=True)
        self.data['time_diff(hr)'].fillna(0, inplace=True)
        choosen_features = ['user_id','purchase_value', 'source','browser','sex','age','ip_address','country','day_of_week','time_diff(hr)','class']
        df_final = self.data[choosen_features]

        return df_final