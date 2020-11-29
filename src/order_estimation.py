from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import gamma
from scipy.special import gammaincc
import matplotlib.pyplot as plt
from functools import partial
import icassp20_T6 as t6
import warnings
import convert_to_mat
import os
import io_functions as io
import pandas as pd
import file_service as fs
import shutil
import mne
import matplotlib.pyplot as plt

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

def run(condition='6'):
    
    col_names = ['subject_name', 'subject_path', 'fsiq', 'cognition_composite_score']
                 
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
    
    evoked_list = []
    
    
    print(subjects_session1)
    for s in subject_list: 
        print(s)
        
        row  = {}
        row['subject_name'] = s[0]
    
        
        row['subject_path'] = s[1]
        row['fsiq'] = s[2]
        row['cognition_composite_score'] = s[3]
        
        subject_name = row['subject_name']
        file_name = os.path.basename(s[1])
        index_of_dot = file_name.index('.')
        file_name_without_extension = file_name[:index_of_dot]
        
        #if subject_name == 'M87196316':
        #    continue
        
        print("Subject " + str(s[0]) + "; Subject Path: " + str(s[1]))
        
        if os.path.isfile(s[1]):
            subject_evoked_data = io.read_evokeds_by_path_and_channel_type(s[1], type='mag',baseline = (-100, 0), condition=condition, verbose=True)
            
            channels_idx = mne.pick_channels(subject_evoked_data.info['ch_names'], include=[], exclude=[])
            groups = dict(all=channels_idx)
            subject_evoked_data_combined = mne.channels.combine_channels(subject_evoked_data, groups, method='mean')
            
            #subject_evoked_data_avg  = subject_evoked_data.average()
            evoked_list.append(subject_evoked_data)
    
            print(file_name_without_extension)
            target_dir = os.path.join(cond_folder_path,subject_name) 
            fs.ensure_dir(target_dir)
            math_lab_file_path = os.path.join(target_dir, file_name_without_extension + '.mat')
            print("Matlab File Path: " + math_lab_file_path)
            
            target_dir_all = cond_folder_path_all
            math_lab_file_path_all = os.path.join(target_dir_all, file_name_without_extension + '.mat')
            
            #save in matlab 
            data = subject_evoked_data_combined.data
            print("Subject Data shape = " + str(data.shape))
            convert_to_mat.numpy_to_mat(data, math_lab_file_path_all)
            convert_to_mat.numpy_to_mat(data, math_lab_file_path)
            row['math_lab_file_path'] = math_lab_file_path
        
            subjects.append(row)
            subject_array.append(data)
            subjects_matlab.append(math_lab_file_path)
            
            row_map = row
        
            subject_map.append(row_map)
            
            #print(subject_evoked_data.info['events'])
            #print(subject_evoked_data.info['meas_id'])
            print(subject_evoked_data.info)
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
    

    
    cond_folder_path_jica = cond_dict_path_jica[condition]
    fs.ensure_dir(cond_folder_path_jica)
    condition_name = cond_name[condition]
    file_name_without_extension  = str(condition_name) + '_' + 'all'
    math_lab_file_path_cond_all = os.path.join(cond_folder_path_jica, file_name_without_extension + '.mat')
    #convert_to_mat.numpy_to_mat(all_subjects_reshaped, math_lab_file_path_cond_all)
    #convert_to_mat.numpy_to_mat(merged.data, math_lab_file_path_cond_all)
    
    subjects = []
    for s in subject_array:
        tc = s.ravel()
        print(tc)
        print(tc.shape)
        subjects.append(tc)
    
    S1 = np.c_[subjects]
    S2 = S1.T
    print(S2.shape)
    
    em_bic = np.array([[1, 1],
                   [2, 2],
                   [2, 4],
                   [3, 3],
                   [3, 4]])
    epsilon = .15
    N_k = 250
    nu = 3
    # Huber
    qH = .8
    # Tukey
    cT = 4.685
    L_max = 15
    
    data, labels, r, N, K_true, mu_true, S_true = t6.data_31(N_k, epsilon)
    
    print(r)
    print(data.shape)
    
    r = S2.shape[1]
    
    igamma = lambda a, b: gammaincc(a, b)* gamma(a)
    cH = np.sqrt(chi2.ppf(qH, r))
    bH = chi2.cdf(cH**2, r+2) + cH**2 / r * (1 - chi2.cdf(cH**2, r))
    aH = gamma(r/2) / np.pi**(r/2) / ( (2*bH)**(r/2) * (gamma(r/2) - igamma(r/2, cH**2 / (2*bH))) + (2*bH*cH**r*np.exp(-cH**2/(2*bH))) / (cH**2 - bH*r))
    g = [partial(t6.g_gaus, r=r),
     partial(t6.g_t, r=r, nu=nu),
     partial(t6.g_huber2, r=r, cH=cH, bH=bH, aH=aH)]

    rho = [partial(t6.rho_gaus, r=r),
       partial(t6.rho_t, r=r, nu=nu),
       partial(t6.rho_huber2, r=r, cH=cH, bH=bH, aH=aH),
       partial(t6.rho_tukey, r=r, cT=cT)]

    psi = [partial(t6.psi_gaus),
       partial(t6.psi_t, r=r, nu=nu),
       partial(t6.psi_huber2, r=r, cH=cH, bH=bH),
       partial(t6.psi_tukey, cT=cT)]

    eta = [partial(t6.eta_gaus),
       partial(t6.eta_t, r=r, nu=nu),
       partial(t6.eta_huber2, r=r, cH=cH, bH=bH),
       partial(t6.eta_tukey, cT=cT)]

    embic_iter = len(em_bic)
    S_est = [[] for _ in range(L_max)]
    mu_est = [[] for _ in range(L_max)]

    bic = np.zeros([embic_iter, L_max, 3])
    like = np.zeros([embic_iter, L_max, 3])
    pen = np.zeros([embic_iter, L_max, 3])
    
    cnt = 1
    for ii_embic in range(embic_iter):
        for ll in range(L_max):
        #EM
            print("Iteration # " + str(cnt))
            mu, S, t, R = t6.EM_RES(S2, ll+1, g[em_bic[ii_embic, 0]-1], psi[em_bic[ii_embic,0]-1])
        
            mu_est[ll].append(mu)
            S_est[ll].append(S)
        
            mem = (R == R.max(axis=1)[:,None])
        
            #BIC        
            bic[ii_embic, ll, 0], like[ii_embic, ll, 0], pen[ii_embic, ll, 0] = t6.BIC_F(data, S_est[ll][ii_embic], mu_est[ll][ii_embic], t, mem,rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])            
            bic[ii_embic, ll, 1], like[ii_embic, ll, 1], pen[ii_embic, ll, 1] = t6.BIC_A(S_est[ll][ii_embic], t, mem, rho[em_bic[ii_embic, 1]-1], psi[em_bic[ii_embic, 1]-1], eta[em_bic[ii_embic, 1]-1])
            bic[ii_embic, ll, 2], like[ii_embic, ll, 2], pen[ii_embic, ll, 2] = t6.BIC_S(S_est[ll][ii_embic], t, mem, rho[em_bic[ii_embic, 1]-1])
            
            print(cnt)
            print(bic[ii_embic, ll, 0], like[ii_embic, ll, 0], pen[ii_embic, ll, 0])
            print(bic[ii_embic, ll, 1], like[ii_embic, ll, 1], pen[ii_embic, ll, 1])
            print(bic[ii_embic, ll, 2], like[ii_embic, ll, 2], pen[ii_embic, ll, 2])
            cnt = cnt +1 

if __name__ == "__main__":
     run('6') # AUDITORY