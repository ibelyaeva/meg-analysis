import numpy as np
import os as os
import pandas as pd
import file_service as fs
import io_functions as io
import csv
import convert_to_mat
import mne
from pathlib import Path
import matplotlib.pyplot as plt

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

org_subjects = {}
org_subjects['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map.csv'
org_subjects['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map.csv'
org_subjects['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map.csv'


bk_subject_path = {}
bk_subject_path['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/reconstructed'
bk_subject_path['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/reconstructed' 
bk_subject_path['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/reconstructed'    

ica_results = {} 
ica_results['6'] = '/pl/meg/analysis/ica_output/processed/aud'
ica_results['4'] = '/pl/meg/analysis/ica_output/processed/vis'
ica_results['2'] = '/pl/meg/analysis/ica_output/processed/aud_vis'

cond_name = {}
cond_name['6'] = 'aud'
cond_name['4'] = 'vis'
cond_name['2'] = 'aud_vis'

def get_subject_name_by_split(name):
    
    subject_name = None
    for n in name:
        if n.startswith('M'):
            subject_name = n
    return subject_name

def read_file(file_path):
    
    row_list = []
    with open(file_path, 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            line = (', '.join(row))
            row_list.append(line)
    return row_list


def get_subject_by_condition(folder, file_path, org_subject_path, cond_name, bk_folder):
    
    subject_list_file_path = os.path.join(folder,file_path)
    subject_list = read_file(subject_list_file_path)
    org_subjects = pd.read_csv(org_subject_path)
   
    print(subject_list)
    cnt = 0
    row_list = []
    
    for s in subject_list:
        row = {}
        cnt = cnt + 1
        print ("Subject Index: " + str(cnt) + "; Subject Path: " + str(s))
        row['subject_index']  = str(cnt)
        subject_prefix = os.path.split(os.path.basename(s))[1]
        print ("Subject Prefix: "  + subject_prefix)
        split = subject_prefix.split('_')
        print ("Subject Prefix Split: "  + str(split))
        subject_name = get_subject_name_by_split(subject_prefix.split('_'))
        print ("Subject Name: "  + str(subject_name))
        
        bk_mat_name = str(cond_name)  + '_ica_br' + str(cnt) + '.mat' 
        bk_mat_path = os.path.join(folder, bk_mat_name)
        ica_mat_name = str(cond_name)  + '_ica_c' + str(cnt) + '-1.mat' 
        ica_mat_path = os.path.join(folder, ica_mat_name)
        
        row['subject_name'] = subject_name
        row['bk_mat_path'] = bk_mat_path
        row['ica_mat_path'] = ica_mat_path    
        row_list.append(row)
        
    subject_ica_df = pd.DataFrame(row_list)
    dataset_id = cond_name + 'subjects_mapping_metadata.csv'
    fig_id = os.path.join(folder,dataset_id)
    subject_ica_df.to_csv(fig_id)
    
    org_subjects = org_subjects.merge(subject_ica_df, how='left', left_on='subject_name', right_on='subject_name')
    bk_meta_fig_id = os.path.join(bk_folder, cond_name + '_bk_meta')
    bk_file_path_csv = bk_meta_fig_id + '.csv'
    org_subjects.to_csv(bk_file_path_csv, index=False)
    print(org_subjects)
    
    return bk_file_path_csv


def run(selected_file_name,condition='6'):
    bk_folder_path = bk_subject_path[condition]
    fs.ensure_dir(bk_folder_path)
    ica_folder = ica_results[condition]    
    org_subjects_path = os.path.join(org_subjects[condition])
    print("org_subjects_path=" + str(org_subjects_path))
    bk_file_meta = get_subject_by_condition(ica_folder, selected_file_name, org_subjects_path, cond_name[condition], bk_folder_path)
    reconctruct(bk_folder_path, bk_file_meta, condition)
    
def get_file_name(file_path):
    return Path(file_path).stem
    
def reconctruct(folder, meta_path, condition):
    bk_subjects = pd.read_csv(meta_path)
    print(bk_subjects.head())
    img_comp_folder = os.path.join(folder, 'fif')
    fs.ensure_dir(img_comp_folder)
    ica_comp_folder = os.path.join(folder, 'ica')
    fs.ensure_dir(ica_comp_folder)
    fig_folder_org = os.path.join(folder, 'fig/org')
    fig_folder_xhat = os.path.join(folder, 'fig/xhat')
    
    fig_folder_ica = os.path.join(folder, 'fig/ica')
    fs.ensure_dir(fig_folder_org)
    fs.ensure_dir(fig_folder_xhat)
    fs.ensure_dir(fig_folder_ica)
    
    for index, s in bk_subjects.iterrows():
        subject_mat_path = s['bk_mat_path']
        subject_fif_path = s['subject_path']
        ica_res = convert_to_mat.mat_numpy(subject_mat_path) 
        ica_set = ica_res['compSet']
        tc = ica_set['timecourse'].item()
        tp = ica_set['topography'].item()
        x_rec = np.dot(tc.T, tp.T)
        
        x_org = io.read_evokeds_by_path_and_channel_type(subject_fif_path, baseline = (-100, 0), condition=condition, verbose=True)
        x_hat = mne.EvokedArray(x_rec, x_org.info, tmin=x_org.times[0], nave=x_org.nave, comment=x_org.comment, verbose=True)
        x_hat_base_file_name = get_file_name(subject_fif_path) + '.fif'
        x_hat_full_path = os.path.join(img_comp_folder,x_hat_base_file_name)
        mne.write_evokeds(x_hat_full_path, x_hat)
                
        fig_id = os.path.join(fig_folder_org,  get_file_name(subject_fif_path) +'.pdf')
        x_org.plot_topomap();
        plt.savefig(fig_id)
        plt.close()
        
        fig_id = os.path.join(fig_folder_xhat,  get_file_name(subject_fif_path) +'.pdf')
        x_hat.plot_topomap();
        plt.savefig(fig_id)
        plt.close()
        
        subject_ica_path = s['ica_mat_path']
        ica_set = convert_to_mat.mat_numpy(subject_ica_path) 
        tc = ica_set['timecourse']
        tp = ica_set['topography']
        x_ica = np.dot(tc.T, tp.T)
        
        x_ica_x_hat = mne.EvokedArray(x_ica, x_org.info, tmin=x_org.times[0], nave=x_org.nave, comment=x_org.comment, verbose=True)
        x_ica_base_file_name = 'ica_' + get_file_name(subject_fif_path) + '.fif'
        x_ica_full_path = os.path.join(ica_comp_folder,x_ica_base_file_name)
        mne.write_evokeds(x_ica_full_path, x_ica_x_hat)
        
        fig_id = os.path.join(fig_folder_ica,  'ica_'  + get_file_name(subject_fif_path) +'.pdf')
        x_ica_x_hat.plot_topomap();
        plt.savefig(fig_id)
        plt.close()
        
        
        print(x_org.info['subject_info'])
        
    
    #ica_res = convert_to_mat.mat_numpy(file_path)
    
if __name__ == "__main__":
    run('audSelectedDataFolders.txt', '6') # AUDITORY
    run('visSelectedDataFolders.txt', '4') # VISUAL
    run('aud_visSelectedDataFolders.txt', '2') # AUD/VISUAL
    
    #run('4') # VISUAL
    #run('2') # AUD/VISUAL