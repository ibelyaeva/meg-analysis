import numpy as np
import csv
import pandas as pd
import os
import file_service as fs
import io_functions as io


all_sessions_demo_beh_scores_path = '/pl/meg/data/meg_ms/MRN/all_sessions_demo_beh_scores_raw.csv'
subjecs_details = '/pl/meg/data/meg_ms/MRN/results/subjects-drop-out'

ne_subjects_all = '/pl/meg/data/meg_ms/NE/metadata/subjects/all_sessions/subjects_ne_all.csv'

def runMRN():
    
    subjects = pd.read_csv(all_sessions_demo_beh_scores_path)
    
    subjects = subjects[subjects['session']=='session3']
    
    
    for index, s in subjects.iterrows():
        subject_name = s['subject_name']
        subject_path = s['subject_path']
        subject = io.read_evokeds_by_path_and_channel_type(subject_path, condition = '6', baseline = (-100, 0), verbose=True)
        s_meta = subject.info
        print ("Meta " + str(s_meta))
        demo = s_meta['subject_info']
        #times = s_meta['times']
        print ("Subject Name: " + str(subject_name) + "; Subject Path " + str(subject_path))
        #print ("Times: " + str(times))
        print ("Demog " + str(demo))
        
    pass

def runNE():
    subjects = pd.read_csv(ne_subjects_all)
    
    for index, s in subjects.iterrows():
        subject_name = s['subject_name']
        subject_path = s['subject_path']
        subject = io.read_evokeds_by_path_and_channel_type(subject_path, condition = '6', baseline = (-100, 0), verbose=True)
        s_meta = subject.info
        print ("Subject Name: " + str(subject_name) + "; Subject Path " + str(subject_path))
        print ("Meta " + str(s_meta))
        print(s_meta['subject_info'])

if __name__ == '__main__':
    #runMRN()
    runNE()