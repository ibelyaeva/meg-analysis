import numpy as np
import convert_to_mat
import os
import io_functions as io
import pandas as pd
import file_service as fs
import shutil
import mne
import matplotlib.pyplot as plt
from scipy import linalg
from numpy import *
import copy
from mne.stats import permutation_t_test
import meg_numeric as mgn
from mne.rank import _estimate_rank_meeg_signals


session1_path = '/pl/meg/data/meg_ms/MRN/session1'
all_sessions_folder = '/pl/meg/data/meg_ms/MRN'
data_path = '/pl/meg/data/meg_ms/MRN/session1'
subject_name = "M87100788_multisensory_session1_all_tsss_mc_defhead_-hp1-lp50-ave.fif"
subject_path = os.path.join(data_path, subject_name)
metadata_path = "/pl/meg/analysis/ica_output"
metadata_name = "subjects_session1_ica_analysis"
session1_metadata_path = "/pl/meg/analysis/subjects/session1_meta.csv"

session1_meta_with_age_path = "/pl/meg/analysis/subjects/session1_meta_with_age.csv"

session1_metadata_matlab_path = "/pl/meg/analysis/subjects/session1_meta_mathlab.csv"

session1_meta_82path = "/pl/meg/data/meg_ms/MRN/results/summary/subjects_82.csv"

cond_name = {'6':'aud',
                  '4':'vis',
                  '2':'aud_vis'
                  }

cond_dict_path = {'6':'/pl/meg/analysis/ica_output/data/session1/aud/subjects',
                  '4':'/pl/meg/analysis/ica_output/data/session1/vis/subjects',
                  '2':'/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects'
                  }

cond_dict_path_all = {'6':'/pl/meg/analysis/ica_output/data/session1/aud/all',
                  '4':'/pl/meg/analysis/ica_output/data/session1/vis/all',
                  '2':'/pl/meg/analysis/ica_output/data/session1/aud_vis/all'
                  }


cond_dict_path_jica = {'6':'/pl/meg/analysis/ica_output/data/session1/aud/jica',
                  '4':'/pl/meg/analysis/ica_output/data/session1/vis/jica',
                  '2':'/pl/meg/analysis/ica_output/data/session1/aud_vis/jica'
                  }

cond_dict_path_merged = {'6':'/pl/meg/analysis/ica_output/data/session1/aud/merged',
                  '4':'/pl/meg/analysis/ica_output/data/session1/vis/merged',
                  '2':'/pl/meg/analysis/ica_output/data/session1/aud_vis/merged'
                  }
    
session1_subject_list = '/pl/meg/analysis/ica_output/data/session1/subject_list.csv'

session1_all_subjects_mat = '/pl/meg/analysis/ica_output/data/session1/subject_list.mat'

array_csv_data = '/pl/meg/analysis/array_data/'

merged_folder_path = '/pl/meg/analysis/ica_output/data/session1/aud/merged'

def merge_evoked(evoked_list, condition):
    merged_evoked = mne.combine_evoked(evoked_list, weights= 'nave')
    cond_folder = cond_dict_path_merged[condition]
    fs.ensure_dir(cond_folder)
    condition_name = cond_name[condition]
    file_name = condition_name + '.fif'
    merged_evoked_file_id = os.path.join(cond_folder, file_name)
    print("merged_evoked shape= " + str(merged_evoked.data.shape))
    print("Writing Merged Evoked for Condition =" + str(cond_name[condition]) + "; @ " + str(merged_evoked_file_id))
    mne.write_evokeds(merged_evoked_file_id, merged_evoked)
    merged_evoked_plot_fig_id = os.path.join(cond_folder, condition_name +  '_' + 'merged_evoked.pdf') 
    merged_evoked_plot_ave_fig_id = os.path.join(cond_folder, condition_name +  '_' + 'merged_evoked_ave.pdf') 
    
    print("Saving figure @ " + str(merged_evoked_plot_fig_id))
    
    merged_evoked_fig = merged_evoked.plot_topomap(times='peaks', time_unit='ms')
    plt.savefig(merged_evoked_plot_fig_id)
    plt.close()

    return merged_evoked
 
def normalize_data(x):
    norm_x = np.linalg.norm(x)
    x_norm = copy.deepcopy(x)
    x_norm = x_norm * (1./norm_x)
    return x_norm
    
def run(condition='6'):
    
    col_names = ['subject_name', 'subject_path', 'fsiq', 'cognition_composite_score', 'age', 'gender']
                 
    #subjects_session1 = pd.read_csv(session1_metadata_path, sep=',', usecols=col_names)
    subjects_session1 = pd.read_csv(session1_meta_82path, sep=',', usecols=col_names)
    cond_folder_path = cond_dict_path[condition]
    cond_folder_path_all = cond_dict_path_all[condition]
    
    fs.ensure_dir(cond_folder_path_all)
    fs.ensure_dir(cond_folder_path)

    subject_list = subjects_session1.values.tolist()
    subjects = []
    
    subjects_matlab = []
    
    
    subject_array = []
    
    subject_map = []
    
    subject_evoked_list = []
    
    info = None
    print(subjects_session1)
    for s in subject_list: 
        print(s)
        
        row  = {}
        row['subject_name'] = s[0]
    
        
        row['subject_path'] = s[1]
        row['fsiq'] = s[2]
        row['cognition_composite_score'] = s[3]
        row['age'] = s[4]
        row['gender'] = s[5]
        
        subject_name = row['subject_name']
        file_name = os.path.basename(s[1])
        index_of_dot = file_name.index('.')
        file_name_without_extension = file_name[:index_of_dot]
        
        #if subject_name == 'M87196316':
        #    continue
        
        layout = None
        montage = None
        channels_names = None
        
        subject_info = None
        print("Subject " + str(s[0]) + "; Subject Path: " + str(s[1]))
        
        if os.path.isfile(s[1]):
            subject_evoked_data = io.read_evokeds_by_path_and_channel_type(s[1], type='mag',baseline = (None, 0), kind='average', condition=condition, verbose=True)
            channels_idx = mne.pick_channels(subject_evoked_data.info['ch_names'], include=[], exclude=[])
            groups = dict(all=channels_idx)
            subject_evoked_data_combined = mne.channels.combine_channels(subject_evoked_data, groups, method='std')
            layout = mne.channels.find_layout(subject_evoked_data_combined.info, ch_type='mag')
            print("Sensor layout =" + str(layout))
            
            montage = subject_evoked_data.get_montage()
            print("Montage layout =" + str(montage))
            
            channels_names = subject_evoked_data.info['ch_names']
            print("Channel Names =" + str(channels_names))
            
            #subject_evoked_data_avg  = subject_evoked_data.average()
            subject_evoked_list.append(subject_evoked_data_combined)
            
            subject_info = subject_evoked_data.info
    
            print(file_name_without_extension)
            target_dir = os.path.join(cond_folder_path,subject_name) 
            fs.ensure_dir(target_dir)
            math_lab_file_path = os.path.join(target_dir, file_name_without_extension + '.mat')
            print("Matlab File Path: " + math_lab_file_path)
            
            fif_file_path = os.path.join(target_dir, file_name_without_extension + '.fif')
            subject_evoked_data_combined.save(fif_file_path)
            print("Saved Subject Combined to @" + str(fif_file_path))
            
            target_dir_all = cond_folder_path_all
            math_lab_file_path_all = os.path.join(target_dir_all, file_name_without_extension + '.mat')
            
            data = subject_evoked_data_combined.data
            data = subject_evoked_data.data
            print("Subject Data shape = " + str(data.shape))
            #save in matlab 
            convert_to_mat.numpy_to_mat(data, math_lab_file_path_all)
            convert_to_mat.numpy_to_mat(data, math_lab_file_path)
            row['math_lab_file_path'] = math_lab_file_path
            row['fif_gfp_file_path'] = fif_file_path
        
            scalings = 1e15
            norm_data = mgn.apply_scaling(data, scalings, verbose=True)
            subjects.append(row)
            subject_array.append(norm_data)
            subjects_matlab.append(math_lab_file_path)
            
            row_map = row
        
            subject_map.append(row_map)
            
            #print(subject_evoked_data.info['events'])
            #print(subject_evoked_data.info['meas_id'])
            print(subject_evoked_data.info)
            info = subject_evoked_data.info
            #print(subject_evoked_data.get_montage())
            #print(subject_evoked_data.info['dig'])
    
            all_subjects = np.dstack(subject_array)
        else:
            print("Subject " + str(s[0]) + "; Subject Path: " + str(s[1]) + "DOES NOT exists")
            

    print(all_subjects.shape)
    subjects_df = pd.DataFrame(subjects)
    subjects_df.to_csv(session1_metadata_matlab_path)
    
    col_names = ['subject_path']
    subjects_matlab_df = pd.DataFrame(subjects_matlab, columns = col_names)
    print(subjects_matlab_df.head())
    
    target_folder = os.path.join(cond_folder_path, 'subject_list.csv')
    subjects_matlab_df.to_csv(target_folder, index=False)
    print("Meta " + str(target_folder))
    
    target_folder = os.path.join(cond_folder_path, 'subject_map.csv')
    subjects_map_df = pd.DataFrame(subject_map)
    subjects_map_df.to_csv(target_folder, index=False)
    print("Subject Meta " + str(target_folder))
    print("All subjects shape:" + str(all_subjects.shape))
    
    N = all_subjects.shape[2]
    C = all_subjects.shape[0]
    T = all_subjects.shape[1]
    all_subjects_reshaped = np.reshape(all_subjects, (N, T*C))
    print("All subjects reshaped shape:" + str(all_subjects_reshaped.shape))
    
    combined_subjects = []
    for s in subject_evoked_list:
        s_data = s.data
        s_dataT = s_data.T
        #print(s_dataT.shape)
        tc = s_dataT.ravel()
        #print(tc)
        #print("Combined tc.shape")
        #print(tc.shape)
        combined_subjects.append(tc)
        
    S_combined = np.c_[combined_subjects]
    print("S_combined.shape= " + str(S_combined.shape))
    
    sfreq = 1000
    ch_names = 82
    channel_types = ['mag']
    nave = C
    
    
    cond_folder_path_jica = cond_dict_path_jica[condition]
    fs.ensure_dir(cond_folder_path_jica)
    condition_name = cond_name[condition]
    create_combined_evoked(merged_folder_path, condition_name, S_combined, sfreq, 0, channels_names, channel_types, nave, layout, montage, subject_info)
    
    file_name_without_extension  = str(condition_name) + '_' + 'all'
    merged_file_name_without_extension  = 'merged_' + str(condition_name) + '_' + 'all'
    merged_condition_path = os.path.join(merged_folder_path, str(condition_name))
    fs.ensure_dir(merged_condition_path)
    merged_file_path = os.path.join(merged_condition_path, merged_file_name_without_extension + '.mat')
    math_lab_file_path_cond_all = os.path.join(cond_folder_path_jica, file_name_without_extension + '.mat')
    #convert_to_mat.numpy_to_mat(all_subjects_reshaped, math_lab_file_path_cond_all)
    convert_to_mat.numpy_to_mat(S_combined, merged_file_path)
    
    subjects = []
    for s in subject_array:
        sT = s.T
        #print(sT.shape)
        tc = s.ravel()
        #print(tc)
        #print("tc.shape")
        #print(tc.shape)
        subjects.append(tc)
    
    S = np.c_[subjects]
    print(S.shape)
    convert_to_mat.numpy_to_mat(S, math_lab_file_path_cond_all)
    print("Saved Subject Matrix to @" + str(math_lab_file_path_cond_all))
    
    
    #rank, sing_val = _estimate_rank_meeg_signals(S.T, info, scalings='norm', tol='auto',return_singular=True, tol_kind='absolute')
    #eigen_values = sing_val**2
    #pve = eigen_values /sum(eigen_values)
    
    #eigen_values_fig_ig = os.path.join(array_csv_data, file_name_without_extension + '.csv')
    #np.savetxt(eigen_values_fig_ig, eigen_values, delimiter=',')
    #print("Saving eigen values to @ " + eigen_values_fig_ig)
    
    #print ("Rank = " + str(rank))
    #print ("Sing Values = " + str(sing_val))

def create_combined_evoked(folder, condition, data, sfreq, tmin, ch_names, channel_types, nave, layout=None, montage=None, subject_info=None):
    
    len_ch = len(ch_names)
    len_sb = data.shape[0]
    del_length =  len_ch - len_sb
    
    for i in range(del_length):
        del(subject_info.ch_names[i])
        
    for i in range(del_length):
        del(subject_info['chs'][i])
    
    print("Channels Length = " + str(len(ch_names)))
    
    print("Subject Info = " + str(subject_info))
    
    subject_info['nchan'] = len(subject_info.ch_names)
          
    
    info = mne.create_info(ch_names, sfreq, ch_types='mag')
    info.set_montage(montage, verbose=True)

    
    evoked_data = mne.EvokedArray(data, subject_info, tmin,kind='standard_error', comment = condition,
                               nave=nave, verbose=True)
    
    evoked_name = 'merged_' +condition + '_all'+ '-ave.fif'
    layout_name = 'merged_' +condition + '_all'+ '-ave.lay'
    evoked_path = os.path.join(folder, evoked_name)
    layout_path = os.path.join(folder, layout_name)
    mne.write_evokeds(evoked_path, evoked_data)
    print("Evoked Path :" + evoked_path)
    
    print("Merged Evoked :" + str(evoked_data))
    print("Merged Evoked Info :" + str(evoked_data.info))
    
    if layout is not None:
        layout.save(layout_path)
        print("Layout Path @:" + layout_path)
        
    

if __name__ == "__main__":
    run('6') # AUDITORY
    run('4') # VISUAL
    run('2') # AUD/VISUAL
    