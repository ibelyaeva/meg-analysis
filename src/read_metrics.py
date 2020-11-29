import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs

mrn_demog_path = '/pl/meg/metadata/csv/nm_dev_cog_demographics.csv'
mrn_scores_path = '/pl/meg/metadata/csv/nm_beh_scores.csv'
session1_path = '/pl/meg/data/meg_ms/MRN/session1'
session2_path = '/pl/meg/data/meg_ms/MRN/session2'
session3_path = '/pl/meg/data/meg_ms/MRN/session3'
session23_path = '/pl/meg/data/meg_ms/MRN/session23'
all_sessions_nm_path = '/pl/meg/data/meg_ms/MRN/subjects_nm.csv'
all_sessions_folder = '/pl/meg/data/meg_ms/MRN'
all_sessions_demo_beh_scores_path = '/pl/meg/data/meg_ms/MRN/all_sessions_demo_beh_scores_raw.csv'

def list_subjects(root, save_dir, prefix, report_name):
    
    cnt = 0
    subject_list = du.get_subjects(root) 
    cnt = len(subject_list)   
    
    rows = []
    row = {}
    
    subject_names = []
    for s in subject_list:
        row = {}
        subject_file = os.path.basename(s)
        subject_name = subject_file.split('_')[0]
        print ("Subject Name: " + str(subject_name))
        subject_names.append(subject_name)
        row['subject_name'] = subject_name
        row['subject_path'] = s
        row['session'] = report_name
        rows.append(row)

    
    subjects_df = pd.DataFrame(rows)
    file_name = report_name + '.csv'
    fig_id = os.path.join(save_dir, file_name)
    fs.ensure_dir(file_name)
    subjects_df.to_csv(fig_id)
    print ("Directory: " + str(root) + 
           "; Subject Count: "  + "; " + str(cnt))
    
    return subjects_df

def get_session(source, target, prefix, report_name):
    list_subjects(source, target, prefix, report_name)
    
def create_session_score_data(file_path):
    score_df = pd.read_csv(file_path)
    return score_df

def read_demog_nm():
    demog_mn = pd.read_csv(mrn_demog_path)   
    print (demog_mn)
    

def read_measures():
    read_demog_nm()
    session1 = list_subjects(session1_path, all_sessions_folder, None, 'session1')
    session2 = list_subjects(session2_path, all_sessions_folder, None, 'session2')  
    session3 = list_subjects(session3_path, all_sessions_folder, None, 'session3')  
    sessions = [session1, session2, session3] 
    all_sessions_df = pd.concat(sessions, ignore_index=True)
    
    file_id = os.path.join(all_sessions_folder, 'all_nm_sessions.csv')
    all_sessions_df.to_csv(file_id)
    
    scores = create_session_score_data(mrn_scores_path)
    
    scores_df = pd.DataFrame() 
    scores_df['subject'] = scores['queried_ursi']
    scores_df['fsiq'] = scores['Session1_WASITWO_014']
    scores_df['cognition_composite_score'] = scores['Session1_PICVOCAB3_021']
    scores_df.drop(scores_df.index[0], inplace=True)
    
    fig_id = os.path.join(all_sessions_folder, 'all_sessions_beh_scores.csv')
    scores_df.to_csv(fig_id)
    
    file_demo_id = os.path.join(all_sessions_folder, mrn_demog_path)
    demo_data = pd.read_csv(file_demo_id)
    
    
    all_sessions_raw_scores = all_sessions_df.copy(True)
    all_sessions_raw_scores = all_sessions_raw_scores.merge(scores_df, how='left', left_on='subject_name', right_on='subject')
    all_sessions_raw_scores = all_sessions_raw_scores.merge(demo_data, how='left', left_on='subject', right_on='URSI')
    all_sessions_raw_scores.drop("URSI", axis=1, inplace=True)
    all_sessions_raw_scores.drop("subject", axis=1, inplace=True)
    all_sessions_raw_scores.rename(columns={"Age": "age", "Gender": "gender", "SES": "ses"})
    all_sessions_raw_scores.to_csv(all_sessions_demo_beh_scores_path)
  
    pass


if __name__ == '__main__':
    read_measures()
    