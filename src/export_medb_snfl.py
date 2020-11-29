import numpy as np
import convert_to_mat
import os
import io_functions as io
import pandas as pd
import file_service as fs
import shutil
import mne
import matplotlib.pyplot as plt

members_path = '/work/design/cloud/snfl-poc/data/snfl_700.txt'
activity_path = '/work/design/cloud/snfl-poc/data/activity_data2.csv'
activity_snfl_path = '/work/design/cloud/snfl-poc/data/member_events_activity.csv'

def run():
    col_names = ['AmisysID']
    member_id_data = pd.read_csv(members_path, sep=',', usecols=col_names)
    activity_data = pd.read_csv(activity_path, sep=',')
    activity_data.drop(activity_data.columns[[29,30]], axis=1, inplace=True)
    #print(member_id_data.head(10))
    print(activity_data.head(10))
    
    activity_data_large = activity_data
    
    N = 35000
    
    
    df_list = []
    for i in range(N):
        activity_data_large = activity_data_large.append(activity_data, ignore_index=False)
        print ("Adding frame")
        print("New Activity Size = " + str(len(activity_data_large)))
        if len(activity_data_large) > 350000:
            break
        

    
    print("New Activity = " + str(len(activity_data_large)))
    activity_data_large.drop(activity_data_large.columns[[0]], axis=1, inplace=True)
    print(activity_data_large.head(10))
    
    activity_data_large['member_id'] = member_id_data['AmisysID']
    print(activity_data_large.head(10))
    activity_data_large.to_csv(activity_snfl_path, index=False)
    print("Saving activity data @ " + str(activity_snfl_path))
    
    print("New Activity Size = " + str(len(activity_data_large)))
    
    
    
    

if __name__ == "__main__":
    run()