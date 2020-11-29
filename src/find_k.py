import kmeans1d

import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs

import pandas as pd
import matplotlib.pyplot as plt
import random
import optimal_cluster

mrn_demog_path = '/pl/meg/metadata/csv/nm_dev_cog_demographics.csv'
mrn_scores_path = '/pl/meg/metadata/csv/nm_beh_scores.csv'
session1_path = '/pl/meg/data/meg_ms/MRN/session1'
session2_path = '/pl/meg/data/meg_ms/MRN/session2'
session3_path = '/pl/meg/data/meg_ms/MRN/session3'
session23_path = '/pl/meg/data/meg_ms/MRN/session23'
all_sessions_nm_path = '/pl/meg/data/meg_ms/MRN/subjects_nm.csv'
all_sessions_folder = '/pl/meg/data/meg_ms/MRN'
iq_score_path = '/pl/meg/data/meg_ms/MRN/subject_iq_scores.csv'
iq_score_clustered_path = '/pl/meg/data/meg_ms/MRN/subject_iq_scores_clustered.csv'


def run():
    file_id = os.path.join(all_sessions_folder, 'beh_scores.csv')
    beh_data = pd.read_csv(file_id)
    beh_data.drop('cognition_composite_score', axis=1, inplace=True)
    beh_data.dropna(axis=0, inplace=True)
    beh_data.to_csv(iq_score_path)
    
    beh_data_df = pd.DataFrame() 
    beh_data_df['fsiq'] = beh_data['fsiq']
    X = beh_data_df.values
    print (X)
    kpp = optimal_cluster.DetK(3,X=X)
    kpp.run(10)
    kpp.plot_all()
    
    pass

if __name__ == '__main__':
    run()