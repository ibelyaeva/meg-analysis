import numpy as np
import os as os
import pandas as pd
import file_service as fs
import io_functions as io
import csv
import convert_to_mat
import mne
from pathlib import Path

from statsmodels.formula.api import ols
from pingouin import pairwise_tukey
import statsmodels.api as sm
import pingouin as pg

import ICAGift as icag

import scipy

from scipy.stats import zscore
from numpy.dual import pinv

org_evoked = {}
org_evoked['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_aud_all-ave.fif'
org_evoked['4'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_vis_all-ave.fif'
org_evoked['2'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_audvis_all-ave.fif'

org_subjects = {}
org_subjects['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map.csv'
org_subjects['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map.csv'
org_subjects['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map.csv'

org_subjects_index = {}
org_subjects_index['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map_index.csv'
org_subjects_index['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map_index.csv'
org_subjects_index['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map_index.csv'   

ica_results_aud5 = {} 
ica_results_aud5['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5'

ica_results_aud10 = {} 
ica_results_aud10['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/10'

ica_results_aud15 = {} 
ica_results_aud15['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/15'

ica_results_aud20 = {} 
ica_results_aud20['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/20'

ica_results_aud25 = {} 
ica_results_aud25['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/25'

# VIS

ica_results_vis2 = {} 
ica_results_vis2['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/2'

ica_results_vis3 = {} 
ica_results_vis3['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/3'

ica_results_vis5 = {} 
ica_results_vis5['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/5'

ica_results_vis10 = {} 
ica_results_vis10['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/10'

ica_results_vis15 = {} 
ica_results_vis15['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/15'

ica_results_vis20 = {} 
ica_results_vis20['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/20'

ica_results_vis25 = {} 
ica_results_vis25['4'] = '/pl/meg/analysis/ica_output/processed_gfp/vis/25'

cond_name = {}
cond_name['6'] = 'aud'
cond_name['4'] = 'vis'
cond_name['2'] = 'aud_vis'

subj_results5 = {} 
subj_results5['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'
subj_results5['4'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'
subj_results5['2'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'

cog_group_path = '/pl/meg/analysis/ica_output/data/all_sessions_demo_beh_scores_processed.csv'
org_subjects_scores_path = '/pl/meg/analysis/ica_output/data/subject_map_with_scores.csv'

import pandasql as ps
import scipy
from scipy.stats import ttest_ind


def generate_with_cog_group(subject_path, cog_path, target_dir):
    subject_df = pd.read_csv(subject_path)
    subject_df_with_scores = subject_df
    subject_list = subject_df_with_scores.values.tolist()
    print("Subject Len " + str(len(subject_df)))
    
    cog_df = pd.read_csv(cog_path)
    cog_df_list = cog_df.values.tolist()
    sql = "select s.*, c.cluster_id, upper(cog_group) as cog_group from subject_df as s inner join cog_df as c on s.subject_name = c.subject_name"
    extended_data_df = ps.sqldf(sql)
    
    score_map = {}
    group_map = {}
    subjects = []
    
    for s in cog_df_list:
        score_map[s[1]] = s[9]
        group_map[s[1]] = s[10]
        # group_map[s[1]] = s[9]
        print(s[1])
        print(s[9])
        print(s[10])
        
    for s in subject_list:
        row = {}
        row['subject_name'] = s[1]
        row['subject_path'] = s[2]
        row['fsiq'] = s[3]
        row['cognition_composite_score'] = s[4]
        row['age'] = s[5]
        row['gender'] = s[6]
        row['math_lab_file_path'] = s[7]
        row['fif_gfp_file_path'] = s[8]
        row['y_hat'] = s[9]
        try:
            row['cluster_id'] = score_map[s[1]]
            row['cog_group'] = group_map[s[1]]
        except:
            print("Not found Cluster Id")
        
        subjects.append(row)
    
    subjects_map_df = pd.DataFrame(subjects)
    print(subjects_map_df)
        
    subject_file_id = os.path.join(target_dir, 'subject_bk_scores.csv')
    subjects_map_df.to_csv(subject_file_id)
    print("Saved Reconstructed Subjects with Cognition Scores @" + subject_file_id)
    return subjects_map_df


def get_condition_name(cond):
    return cond_name[cond]


def get_matlab_solution_path(target_dir, cond_name, comp_num):
    subject_group_name = cond_name + '_' + str(comp_num) + '_ica_br1' + '.mat' 
    subject_group_path = os.path.join(target_dir, subject_group_name)
    return subject_group_path


def get_matlab_solution_path_ica(target_dir, cond_name, comp_num):
    subject_group_name = cond_name + '_' + str(comp_num) + '_ica_c1-1' + '.mat' 
    subject_group_path = os.path.join(target_dir, subject_group_name)
    return subject_group_path


def get_tc(map_name, comp, cond):
    
    condition_name = get_condition_name(cond)
    ica_path = map_name[cond]
    file_name = 'tc' + '_' + condition_name + '_' + str(comp) + '.npy'
    tc_path = os.path.join(ica_path, 'np', file_name)
    tc = np.load(tc_path)
    return tc


def get_tc_by_path(tc_path):
    tc = np.load(tc_path)
    return tc


def get_tp_by_path(tp_path):
    tp = np.load(tp_path)
    return tp


def get_tp(map_name, comp, cond):
    
    condition_name = get_condition_name(cond)
    ica_path = map_name[cond]
    file_name = 'tp' + '_' + condition_name + '_' + str(comp) + '.npy'
    tp_path = os.path.join(ica_path, 'np', file_name)
    tp = np.load(tp_path)
    return tp


def get_subject__data(map_name):
    subject_data = pd.read_csv(map_name)
    
    return subject_data    


def get_subject_res_folder(folder_path):
    
    csv_folder = os.path.join(folder_path, 'csv')
    fs.ensure_dir(csv_folder)
    return csv_folder


def save_tc(target_dir, cond_name, x, comp_num):
    output_folder = os.path.join(target_dir, 'subj/data/agg')
    fs.ensure_dir(output_folder)
    tc_condition_file_id = 'tc_' + cond_name + '_' + str(comp_num) + '.npy'
    tc_condition_fig_id = os.path.join(output_folder, tc_condition_file_id)
    np.save(tc_condition_fig_id, x)
    
    print ("Saved TC @ " + tc_condition_fig_id)
    
    return tc_condition_fig_id

    
def save_tp(target_dir, cond_name, x, comp_num):
    output_folder = os.path.join(target_dir, 'subj/data/agg')
    fs.ensure_dir(output_folder)
    tp_condition_file_id = 'tp_' + cond_name + '_' + str(comp_num) + '.npy'
    tp_condition_fig_id = os.path.join(output_folder, tp_condition_file_id)
    np.save(tp_condition_fig_id, x)
    
    print ("Saved TP @ " + tp_condition_fig_id)
    
    return tp_condition_fig_id


def save_agg_solution(target_dir, cond_name, comp_num):
    
    subject_matlab_path = get_matlab_solution_path(target_dir, cond_name, comp_num)
    bk_mat_path = os.path.join(subject_matlab_path)
    bk = convert_to_mat.mat_numpy(bk_mat_path)
    bk_set = bk['compSet']
    tc_bk = bk_set['timecourse'].item()
    tp_bk = bk_set['topography'].item().T
    
    tc_path = save_tc(target_dir, cond_name, tc_bk, comp_num)
    tp_path = save_tp(target_dir, cond_name, tp_bk, comp_num)
    
    return tc_path, tp_path


def reconstruct_subject_tc(tc_path, tp_path, k, target_dir):
    tc = get_tc_by_path(tc_path)
    tp = get_tp_by_path(tp_path)
    w = np.linalg.pinv(tc)
    print ("w org shape" + str(w.shape))
    tc = tc.T
    print("tcT.shape = " + str(tc.shape))
    print("tpT.shape = " + str(tp.T.shape))
    print("tc[k]T.shape = " + str(tc[k].T.shape))
    y_k = np.dot(tc[k, :], tp.T)
    print ("y[k] shape" + str(y_k.shape))
    print ("Subject # " + str(k))
    y_k_path = os.path.join(target_dir, 's_' + str(k + 1) + '.npy')
    np.save(y_k_path, y_k)
    a_k = tc[:, 2]
    w = np.linalg.pinv(tc)
    print ("w shape" + str(w.shape))
    w_k = w[2, :]
    print ("a_k shape" + str(a_k.shape))
    print ("w_k shape" + str(w_k.shape))
    C = np.dot(w, tc)
    print ("c shape" + str(C.shape))
    print ("c =" + str(C))
    print (y_k_path)
    return y_k_path   


def reconstruct_subject_source(tc_path, k, target_dir, x):
    tc = get_tc_by_path(tc_path)
    w = np.linalg.pinv(tc)
    w_k = w[k, :]
    w_k_1D = w_k[:, np.newaxis]
    print("w_k_1D shape = " + str(w_k_1D.shape))
    x_k = zscore(x[k])
    x_k_1D = x_k[:, np.newaxis]
    print("x_k_1D.shape = " + str(x_k_1D.shape))
    y_k = np.dot(w_k_1D, x_k_1D.T)
    print("y_k.shape = " + str(y_k.shape))
    y_k_path = os.path.join(target_dir, 'y_' + str(k + 1) + '.npy')
    print (y_k_path)
    np.save(y_k_path, y_k)
    return y_k_path     

    
def reconstruct_subject(source_subj, target_dir, tc_path, tp_path, cond, evoked_path):
    output_folder = os.path.join(target_dir, 'subj/source')
    fs.ensure_dir(output_folder)
    csv_folder = os.path.join(target_dir, 'subj/csv')
    fs.ensure_dir(csv_folder)
    subject_fig_id = os.path.join(csv_folder, 'subject_sources.csv')
    fs.ensure_dir(csv_folder)
    subjects = pd.read_csv(source_subj, sep=',')
    subjects_df = subjects
    evoked_fif = io.read_evokeds_by_path_and_channel_type_singles(evoked_path, baseline=(None, 0), verbose=True, kind='standard_error')
    print ("Evoked fif = " + str(evoked_fif[0]))
    evoked_data = evoked_fif[0].data
    print ("Evoked data shape= " + str(evoked_data.shape))
    
    cnt = 0
    subject_list = subjects.values.tolist()
    rows = []
    for s in subject_list:
        row = {}
        y_k_path = reconstruct_subject_source(tc_path, cnt, output_folder, evoked_data)
        cnt = cnt + 1
        row['y_hat_path'] = y_k_path
        rows.append(y_k_path)
        
    subjects_df['y_hat_path'] = rows
    subjects_df.to_csv(subject_fig_id)
    print("Saved Subjects Sources @" + subject_fig_id)
    generate_with_cog_group(subject_fig_id, cog_group_path, csv_folder)


def reconstruct_by_condition(source, target_dir, cond, comp_num, evoked_path):
    condition_name = get_condition_name(cond)
    output_folder = os.path.join(target_dir, 'results')
    fs.ensure_dir(output_folder)
    tc_path, tp_path = save_agg_solution(target_dir, condition_name, comp_num)
    reconstruct_subject(source, target_dir, tc_path, tp_path, cond, evoked_path)


def reconstruct_aud():
    reconstruct_by_condition(org_subjects['6'], ica_results_aud5['6'], '6', 5, org_evoked['6'])
    reconstruct_by_condition(org_subjects['6'], ica_results_aud10['6'], '6', 10, org_evoked['6'])
    reconstruct_by_condition(org_subjects['6'], ica_results_aud15['6'], '6', 15, org_evoked['6'])
    reconstruct_by_condition(org_subjects['6'], ica_results_aud20['6'], '6', 20, org_evoked['6'])
    reconstruct_by_condition(org_subjects['6'], ica_results_aud25['6'], '6', 25, org_evoked['6'])

    
def reconstruct_vis():
      
    reconstruct_by_condition(org_subjects['4'], ica_results_vis2['4'], '4', 2, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis3['4'], '4', 3, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis5['4'], '4', 5, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis10['4'], '4', 10, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis15['4'], '4', 15, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis20['4'], '4', 20, org_evoked['4'])
    reconstruct_by_condition(org_subjects['4'], ica_results_vis25['4'], '4', 25, org_evoked['4'])


if __name__ == "__main__":
    comp_num = 5
    reconstruct_aud()
    reconstruct_vis()
