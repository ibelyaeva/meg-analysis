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
from mne.stats import permutation_t_test

from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)

import ICAGift as icag

org_subjects = {}
org_subjects['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map.csv'
org_subjects['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map.csv'
org_subjects['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map.csv'


bk_subject_path = {}
bk_subject_path['6'] = '/pl/meg/analysis/ica_output/data/session1/jica/aud/reconstructed'
bk_subject_path['4'] = '/pl/meg/analysis/ica_output/data/session1/jica/vis/reconstructed' 
bk_subject_path['2'] = '/pl/meg/analysis/ica_output/data/session1/jica/aud_vis/reconstructed'    

ica_results = {} 
ica_results['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5'
ica_results['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/5'
ica_results['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/5'

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
    #print("org_subjects_path=" + str(org_subjects_path))
    #bk_file_meta = get_subject_by_condition(ica_folder, selected_file_name, org_subjects_path, cond_name[condition], bk_folder_path)
    reconctruct(bk_folder_path, None, condition)
    
def get_file_name(file_path):
    return Path(file_path).stem
    
def reconctruct(folder, x_path, condition, comp_num):
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
       
    np_comp_folder = os.path.join(folder, 'np')
    fs.ensure_dir(np_comp_folder)
    
    subj_comp_folder = os.path.join(folder, 'subj')
    fs.ensure_dir(subj_comp_folder)
   
    condition_name = cond_name[condition]
    bk_mat_name = condition_name  + '_' + str(comp_num) + '_ica_br1' + '.mat' 
    bk_mat_path = os.path.join(folder, bk_mat_name)
    
    ica_mat_name = str(condition_name)  + '_' +str(comp_num) + '_ica_c1' + '-1.mat' 
    ica_mat_path = os.path.join(folder, ica_mat_name)
    
    ica_res = convert_to_mat.mat_numpy(bk_mat_path) 
    ica_set = ica_res['compSet']
    tc_bk = ica_set['timecourse'].item()
    tp_bk = ica_set['topography'].item()
    x_rec = np.dot(tc_bk.T, tp_bk)
    
    # reconstructed subject
    x_org = io.read_evokeds_by_path_and_channel_type_singles(x_path, baseline = (None, 0),verbose=True, kind='standard_error')

    x_hat = mne.EvokedArray(x_rec, x_org[0].info, tmin=x_org[0].times[0], nave=x_org[0].nave, comment=x_org[0].comment, verbose=True)
    x_hat_base_file_name = 'x_hat_' + condition_name
    x_hat_fif_name = x_hat_base_file_name + '.fif'
    x_hat_full_path = os.path.join(img_comp_folder,x_hat_fif_name)
    
    times = x_org[0].times
    print ("Times = " + str(times))

    #ica
    
    subject_ica_path = ica_mat_path
    ica_set = convert_to_mat.mat_numpy(subject_ica_path) 
    tc = ica_set['timecourse']
    tp = ica_set['topography']
   
    print("x ica tc.shape" + str(tc.shape))
    print("x ica tp.shape" + str(tp.shape))
    
    x_ica_info = x_org[0].info
      
    x_ica_base_file_name = 'ica_' + condition_name + '_' +str(tc.shape[0])
    fig_id = os.path.join(fig_folder_ica,  x_ica_base_file_name)
    x_tc_base_file_name = 'tc_' + condition_name + '_' +str(tc.shape[0])
    fig_id_tc = os.path.join(fig_folder_ica,  x_tc_base_file_name)
    
    ica = icag.ICA(tc, x_ica_info, tp.T, times)
    ica.plot_components(fig_id=fig_id)
    ica.plot_ica_component(fig_id=fig_id)
    ica.plot_tc(condition_name.upper(),fig_id=fig_id_tc)
    
    cnt = 0
    for x in tc:
        #print("x_data" + str(x.flatten()))
        #mne.viz.plot_topomap(x,x_ica_info,ch_type='mag')
        cnt = cnt + 1
    
    ch_names = comp_num
    sfreq =1000
    info = mne.create_info(ch_names, sfreq, ch_types='mag')
    x_ica_x_hat = mne.EvokedArray(tp.T, info, tmin=0, nave=tc.shape[1], verbose=True)
    
    x_ica_base_file_name = 'ica_' + condition_name
    x_ica_base_file_fif  = x_ica_base_file_name + '.fif'
    x_ica_full_path = os.path.join(ica_comp_folder,x_ica_base_file_fif)
    mne.write_evokeds(x_ica_full_path, x_ica_x_hat)
    
    tc_condition_file_id = 'tc_' + condition_name + '_' + str(comp_num) + '.npy'
    tc_condition_fig_id = os.path.join(np_comp_folder, tc_condition_file_id)
    np.save(tc_condition_fig_id, tc)
    
    print ("Saved TC @ " + tc_condition_fig_id)
    
    tp_condition_file_id = 'tp_' + condition_name + '_' + str(comp_num) + '.npy'
    tp_condition_fig_id = os.path.join(np_comp_folder, tp_condition_file_id)
    
    np.save(tp_condition_fig_id, tp)
    
    print ("Saved TP @ " + tp_condition_fig_id + '_' + str(tp.shape))
    
    #bk subject tc
    
    subj_comp_folder_data = os.path.join(subj_comp_folder, 'data')
    fs.ensure_dir(os.path.join(subj_comp_folder_data))
                  
    tc_condition_file_id = 'tc_' + condition_name + '_' + str(comp_num) + '.npy'
    tc_condition_fig_id = os.path.join(subj_comp_folder_data, tc_condition_file_id)
    np.save(tc_condition_fig_id, tc_bk)
    
    print ("Saved Subject TC @ " + tc_condition_fig_id)
    
    tp_condition_file_id = 'tp_' + condition_name + '_' + str(comp_num) + '.npy'
    tp_condition_fig_id = os.path.join(subj_comp_folder_data, tp_condition_file_id)
    
    np.save(tp_condition_fig_id, tp_bk)
    
    print ("Saved TP @ " + tp_condition_fig_id + '_' + str(tp.shape))
    
    x_ica_base_file_name = 'ica_' + condition_name + '_' +str(tc.shape[0])
    fig_id = os.path.join(subj_comp_folder,  x_ica_base_file_name)
    x_tc_base_file_name = 'tc_' + condition_name + '_' +str(tc.shape[0])
    fig_id_tc = os.path.join(subj_comp_folder,  x_tc_base_file_name)
    
    ica = icag.ICA(tc_bk.T, x_ica_info, tp_bk, times)
    #ica.plot_components(fig_id=fig_id)
    #ica.plot_ica_component(fig_id=fig_id)
    ica.plot_tc(condition_name.upper(),fig_id=fig_id_tc)
    
    

if __name__ == "__main__":
    
    ica_folder_aud_5 = '/pl/meg/analysis/ica_output/processed_gfp/aud/5'
    aud_path = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_aud_all-ave.fif'
    
    ica_folder_aud_25 = '/pl/meg/analysis/ica_output/processed_gfp/aud/25'
    
    meg_layout = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_aud_all-ave.lay'
    
    reconctruct(ica_folder_aud_5, aud_path, '6', 5) # AUDITORY
    #run('visSelectedDataFolders.txt', '4') # VISUAL
    #run('aud_visSelectedDataFolders.txt', '2') # AUD/VISUAL
    
    #run('4') # VISUAL
    #run('2') # AUD/VISUAL