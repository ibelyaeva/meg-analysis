import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs
import shutil

root =  "/pl/meg/data/meg_ms/NE"
data_root = "/pl/meg/data/meg_ms/NE/all"
group1_path = os.path.join(data_root, "group1")
group2_path = os.path.join(data_root, "group2")
metadata_root = "/pl/meg/data/meg_ms/NE/metadata/subjects"
metadata_stat = "/pl/meg/data/meg_ms/NE/metadata/subjects/stat"

session1_path = os.path.join(root, "session1")
session2_path = os.path.join(root, "session2")
session3_path = os.path.join(root, "session3")

session1_folder = os.path.join(metadata_root, "session1")
session2_folder  = os.path.join(metadata_root, "session2")
session3_folder  = os.path.join(metadata_root, "session3")
session_folder_all  = os.path.join(metadata_root, "all_sessions")

fs.ensure_dir(session1_path)
fs.ensure_dir(session2_path)
fs.ensure_dir(session3_path)

fs.ensure_dir(session1_folder)
fs.ensure_dir(session2_folder)
fs.ensure_dir(session3_folder)
fs.ensure_dir(session_folder_all)
fs.ensure_dir(metadata_stat)

session_dict = {}
session_dict['visit1'] = 'session1'
session_dict['visit2'] = 'session2'
session_dict['visit3'] = 'session3'

def list_subjects(root_folder, save_dir, search_session, report_name, target_dir):
    
    cnt = 0
    subject_list = du.get_subjects(root_folder) 
    cnt = len(subject_list)   
    
    rows = []
    row = {}
    
    subject_names = []
    for s in subject_list:
        row = {}
        subject_file = os.path.basename(s)
        print("Subject File Path: " + str(subject_file))
        subject_name = subject_file.split('_')[0]
        session = subject_file.split('_')[2]
        
        if session == search_session:
            row['visit'] = search_session
            row['session'] = session_dict[search_session]
            print ("Subject Name: " + str(subject_name))
            
            subject_names.append(subject_name)
            row['subject_name'] = subject_name
            row['subject_path'] = s
            rows.append(row)
            print("Target Dir: " + target_dir)
            copy_to(target_dir, subject_name, s)

    
    subjects_df = pd.DataFrame(rows)
    file_name = report_name + '.csv'
    fig_id = os.path.join(save_dir, file_name)
    fs.ensure_dir(file_name)
    subjects_df.to_csv(fig_id)
        
    print ("Directory: " + str(root) + 
           "; Subject Count: "  + "; " + str(cnt))
    
    return subjects_df

def copy_to(target_dir, subject_name, subject_path):
    
        subject_file = os.path.basename(subject_path)
        destination_path = os.path.join(target_dir, subject_file)
        print("Source Path : " + subject_path + "; Target Path: " + destination_path)
        print ("Copy " + subject_name + '->' + destination_path)
        shutil.copy(subject_path, destination_path)

def read_measures():
   
    session1_fig_id = "subjects_ne_session1.csv"
    session2_fig_id = "subjects_ne_session2.csv"
    session3_fig_id = "subjects_ne_session3.csv"
       
    session1 = list_subjects(data_root, session1_folder, "visit1", session1_fig_id, session1_path)
    session2 = list_subjects(data_root, session2_folder, "visit2", session2_fig_id, session2_path)
    session3 = list_subjects(data_root, session3_folder, "visit3", session3_fig_id, session3_path)
       
    print("Session1 Subject Count = " + str(len(session1)))
    print("Session2 Subject Count = " + str(len(session2)))
    print("Session3 Subject Count = " + str(len(session3)))
    
    sessions = [session1, session2, session3] 
    all_sessions_df = pd.concat(sessions, ignore_index=True)
    
    session_all_fig_id = "subjects_ne_all.csv"
    session_all_fig_id_path = os.path.join(session_folder_all, session_all_fig_id)
    all_sessions_df.to_csv(session_all_fig_id_path)
    
    
    print("Total Subject Count = " + str(len(all_sessions_df)))

def get_session(source, target, prefix, report_name):
    list_subjects(source, target, prefix, report_name)


if __name__ == '__main__':
    read_measures()