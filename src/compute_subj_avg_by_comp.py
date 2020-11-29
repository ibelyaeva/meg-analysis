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
import matplotlib.pyplot as plt
import scipy
from mne.stats import permutation_cluster_test
from scipy.stats import zscore
import ICA as ica
import save_plots_util as plt_util
from mne.channels import equalize_channels
import pandasql as ps

import meg_contrast as mc
import ICAContrast as ica_c

aud_path = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_aud_all-ave.fif'
vis_path = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_vis_all-ave.fif'
aud_vis_path = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_audvis_all-ave.fif'

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

org_evoked = {}
org_evoked['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_aud_all-ave.fif'
org_evoked['4'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_vis_all-ave.fif'
org_evoked['2'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_audvis_all-ave.fif'

vis_layout = {}
vis_layout['4'] = '/pl/meg/analysis/ica_output/data/session1/aud/merged/merged_vis_all-ave.lay'


from mne.stats import permutation_t_test
from scipy import stats

def reconstruct_evoked(x, data, cond):
    
    print("Evoked = " + str(data.shape))
    
    sfreq = 1000
    s = x[0]
    montage = s.get_montage()
    print("Montage = " + str(s.get_montage()))
    subject_info = s.info
    
    print(s.info['events'])
    
    print("Channels Name = " + str(subject_info.ch_names))
    len_ch = len(subject_info.ch_names)
    len_sb = data.shape[0]
    del_length =  len_ch - len_sb
    add_lengh = data.shape[0] 
    
    print("To delete Channels = " + str(del_length))
    
    cnt = 0
    channel_names = []
    for c in subject_info.ch_names:
        if (cnt < add_lengh):
            channel_names.append(c)
        cnt = cnt + 1
        
    chs = []
    cnt = 0
    for i in subject_info['chs']:
        if (cnt < add_lengh):
            chs.append(i)
        cnt = cnt + 1

    
    print("Channels Length = " + str(len(channel_names)))
    subject_info['ch_names'] = channel_names
    subject_info['chs'] = chs
      
    subject_info['nchan'] = data.shape[0]  
    info = mne.create_info(channel_names, sfreq, ch_types='mag')
    info.set_montage(montage, verbose=True)
    
    print("Subject Info = " + str(subject_info))
    evoked_data = mne.EvokedArray(data, subject_info, s.tmin, kind='standard_error', comment = cond,
                               nave=1, verbose=True)
    
    print(evoked_data)
    
    return evoked_data
    
def get_file_name(file_path):
    return Path(file_path).stem

def generate_info():
    pass

def generate_evoked(fif_file_path, data, cond):
    x_org = io.read_evokeds_by_path_and_channel_type_singles(fif_file_path, baseline = (None, 0),verbose=True, kind='standard_error')
    evoked  = reconstruct_evoked(x_org, data, cond)
    return evoked

def get_condition_name(cond):
    return cond_name[cond]

def get_tp_by_path(target_dir, condition_name, comp_num):
    tp_path = os.path.join(target_dir, 'subj/data/agg/' + 'tp_' +condition_name + '_' + str(comp_num) + '.npy')
    tp = np.load(tp_path).T
    print("Reading TP @ " + tp_path + "; TP.shape: " + str(tp.shape))
    return tp

def get_tc_by_path(target_dir, condition_name, comp_num):
    tc_path = os.path.join(target_dir, 'subj/data/agg/' + 'tc_' +condition_name + '_' + str(comp_num) + '.npy')
    tc = np.load(tc_path)
    print("Reading TC @ " + tc_path + "; TC.shape: " + str(tc.shape))
    return tc

def compute_significant_time(x, n_perm, times, tmin=0, tmax=0.008):
    
    print(x.shape)
    temporal_mask = np.logical_and(tmin <= times, times <=  tmax)
    #x = x[:, temporal_mask]
    print(x.shape)
    #x = np.reshape(x, (1, x.shape[0]))
    t, p_values, p = permutation_t_test(x, n_perm, n_jobs=1)
    print(t, p_values)
    #time_inds = times[p_values <= 0.05]
    #sig_times =  times[time_inds]
    #print("Significant Times = " + str(sig_times))
    
    
def compute_clusters(x, n_perm, times, tmin=0, tmax=0.008, fig_id = None):
    
    print(x.shape)
    x_compare = np.zeros_like(x)
    data = [x, x_compare]
    threshold = 3
    t_obs, clusters, cluster_p_values, h0 = permutation_cluster_test(data, n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=1, verbose=True)
    
    print(cluster_p_values)
    for i_c, c in enumerate(clusters):
        c = c[0]
        print(c)
        if cluster_p_values[i_c] <= 0.05:
            h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
            print("start = " + str(times[c.start]))
            print("stop = " + str(times[c.stop - 1]))
        else:
            plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)
    hf = plt.plot(times, t_obs, 'g')
    #plt.legend((h, ), ('cluster p-value < 0.05', ))
    plt.xlabel("time (ms)")
    plt.ylabel("f-values")
    
    if fig_id is not None:
        plt_util.save_fig_pdf_no_dpi(fig_id)
     

   
    
def find_siginificant_times(data, n_perm, times, fig_id=None):
    
    #data = data.T
    #compute_significant_time(data, n_perm, times)
    
    
    compute_clusters(data,n_perm, times, tmin=0, tmax=0.008, fig_id=fig_id)

def create_info(self, data, evoked):
        
        ch_names = data.shape[0]
        info = mne.create_info(ch_names, 1000, ch_types='mag')
        add_lengh = data.shape[0] - len(self.info['ch_names'])
        
        print("Channels Name = " + str(self.info['ch_names']))
        
        cnt = 0
        channel_names = []
        for c in self.info['ch_names']:
            if (cnt < add_lengh):
                channel_names.append(c)
            cnt = cnt + 1
        
        info = mne.create_info(channel_names, 1000, ch_types='mag')
        info.set_montage(evoked.get_montage(), verbose=True)
        
        print("Montage = " + str(info))
        
        return info
    
def create_group_df(data, group_name):
    pass
  
def generate_results_by_component(target_dir, cond, comp_num, col_num, cols=2, figsize=(15,5)):
    condition_name = get_condition_name(cond)
    subj_folder = os.path.join(target_dir, 'subj', 'csv')
    subj_stat_folder = os.path.join(target_dir, 'subj', 'csv', 'stat')
    subj_group_folder = os.path.join(target_dir, 'subj/group')
    fs.ensure_dir(subj_group_folder)
    fs.ensure_dir(subj_stat_folder)

    fs.ensure_dir(subj_folder)
    fig_folder = os.path.join(target_dir, 'subj', 'fig')
    fs.ensure_dir(fig_folder)
    subj_fig_id = os.path.join(target_dir, 'subj', 'csv', 'subject_bk_scores.csv')
    subjects_df = pd.read_csv(subj_fig_id)
    subject_list = subjects_df.values.tolist()
    subject_tc = get_tc_by_path(target_dir, condition_name, comp_num)
    subject_tp = get_tp_by_path(target_dir, condition_name, comp_num)
    
    agg_ica_fig_id = os.path.join(fig_folder, condition_name + '_agg_' + str(comp_num) + '_component_' + str(col_num))
    agg_ica_tc_grid_fig_id = os.path.join(fig_folder, condition_name + '_tc_grid_agg_' + str(comp_num) + '_component_' + str(col_num))
    cluster_fig_id = os.path.join(fig_folder, condition_name + '_tc_cluster_' + str(comp_num) + '_component_' + str(col_num))
    subject_tp_evoked = generate_evoked(org_evoked[cond], subject_tp, cond)
    
    ic_name = "IC" + str(col_num)
    evoked_component = mc.Contrast(ic_name, subject_tp[col_num-1], subject_tp_evoked, subject_tp_evoked.times)
    
    high_group_list = []
    med_group_list = []
    low_group_list = []
    
    for s in subject_list:
        print(s)
        x_hat_path = s[9]
        print("x_hat_path = " + str(x_hat_path))
        x_hat = np.load(x_hat_path)
        print("subject_shape = " + str(x_hat.shape))
        print ("X HAT")
        print (x_hat)
        print ("X HAT")
          
        cog_group = s[11]
        
        if cog_group == 'High':
            add = x_hat[col_num-1]
            high_group_list.append(x_hat[col_num-1])
            print("high_group subject_shape = " + str(add.shape))
            
        if cog_group == 'Medium':
            med_group_list.append(x_hat[col_num-1])
            
        if cog_group == 'Low':
            low_group_list.append(x_hat[col_num-1])
    
    #compute all needed contrast
    
    high_group_ds = np.array(high_group_list)
    med_group_ds = np.array(med_group_list)
    low_group_ds = np.array(low_group_list)
    
    mean_high_ds = np.mean(high_group_list, axis = 0)
    mean_high_ds = mean_high_ds[:, np.newaxis].T  
    mean_high_ds = np.vstack((mean_high_ds, mean_high_ds))
    
    mean_med_ds = np.mean(med_group_list, axis = 0)
    mean_med_ds = mean_med_ds[:, np.newaxis].T
    mean_med_ds = np.vstack((mean_med_ds, mean_med_ds))
    
    mean_low_ds = np.mean(low_group_list, axis = 0)
    mean_low_ds = mean_low_ds[:, np.newaxis].T
    mean_low_ds = np.vstack((mean_low_ds, mean_low_ds))
    
    high_group_comp_evoked = generate_evoked(org_evoked[cond], mean_high_ds, cond)
    med_group_comp_evoked = generate_evoked(org_evoked[cond],  mean_med_ds, cond)
    low_group_comp_evoked = generate_evoked(org_evoked[cond],  mean_low_ds, cond)
    
    high_group_comp_evoked_file_id = os.path.join(subj_group_folder, 'high_group_comp_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    high_group_comp_evoked.save(high_group_comp_evoked_file_id)
    high_group_comp_component = mc.Contrast("High IQ IC",mean_high_ds, high_group_comp_evoked, high_group_comp_evoked.times)
    
    med_group_comp_evoked_file_id = os.path.join(subj_group_folder, 'med_group_comp_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    med_group_comp_evoked.save(med_group_comp_evoked_file_id)
    med_group_comp_component = mc.Contrast("Medium IQ IC",mean_med_ds , med_group_comp_evoked, med_group_comp_evoked.times)
    
    low_group_comp_evoked_file_id = os.path.join(subj_group_folder, 'low_group_comp_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    low_group_comp_evoked.save(low_group_comp_evoked_file_id)
    low_group_comp_component = mc.Contrast("Low IQ IC",mean_low_ds, low_group_comp_evoked, low_group_comp_evoked.times)
    
    print ("high_group_ds = " + str(len(high_group_ds)))
    print ("med_group_ds = " + str(len(med_group_ds)))
    print ("low_group_ds = " + str(len(low_group_ds)))
    
    stat = []
    stat_row = {}
    stat_row['high_group_ds'] = len(high_group_ds)
    stat_row['med_group_ds'] = len(med_group_ds)
    stat_row['low_group_ds'] = len(low_group_ds)
    stat.append(stat_row)
    
    cog_stat_df = pd.DataFrame(stat)
    cog_stat_df_file_id = os.path.join(subj_group_folder, 'cog_stat_df_82_session1.csv')
    cog_stat_df.to_csv(cog_stat_df_file_id)
    print("Saved cog_stat_df @ " + str(cog_stat_df_file_id))
    
    col_cog_group_id = 'cog_group_column_group_summary_' + str(col_num) + '.csv'
    col_cog_group_summary_id = os.path.join(subj_stat_folder, col_cog_group_id)
     
    col_cog_group_df  = pd.read_csv( col_cog_group_summary_id)
    col_cog_group_df['value'] = col_cog_group_df['group']
    high_group = ps.sqldf("select tc from col_cog_group_df where value == 'High'")
    low_group = ps.sqldf("select tc  from col_cog_group_df where value == 'Low'")
    med_group = ps.sqldf("select tc from col_cog_group_df where value == 'Medium'") 
     
    high_vs_low_contrast = pd.concat([high_group,low_group], axis=1)
    high_vs_low_contrast = high_vs_low_contrast.dropna()
    high_vs_low_evoked = generate_evoked(org_evoked[cond], high_vs_low_contrast.values, cond)
    
    high_vs_low_evoked_file_id = os.path.join(subj_group_folder, 'sp_filter_high_vs_low_evoked_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    high_vs_low_evoked.save(high_vs_low_evoked_file_id)
    high_vs_low_group_comp_contrast = mc.Contrast("High > Low",high_vs_low_contrast.values, high_vs_low_evoked, high_vs_low_evoked.times)
    
    high_vs_med_contrast = pd.concat([high_group,med_group], axis=1)
    high_vs_med_contrast = high_vs_med_contrast.dropna()
    
    high_vs_med_evoked = generate_evoked(org_evoked[cond], high_vs_med_contrast.values, cond)
    high_vs_med_evoked_file_id = os.path.join(subj_group_folder, 'sp_filter_high_vs_med_evoked_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    high_vs_low_evoked.save(high_vs_med_evoked_file_id)
    high_vs_med_group_comp_contrast = mc.Contrast("High > Medium",high_vs_med_contrast.values, high_vs_med_evoked, high_vs_med_evoked.times)
       
    low_vs_med_contrast = pd.concat([low_group,med_group], axis=1)
    low_vs_med_contrast = low_vs_med_contrast.dropna()
    
    low_vs_med_evoked = generate_evoked(org_evoked[cond], low_vs_med_contrast.values, cond)
    low_vs_med_evoked_file_id = os.path.join(subj_group_folder, 'sp_filter_low_vs_med_evoked_K_' + str(comp_num) + '_colnum_' + str(col_num) + '_' + condition_name) + '.fif'
    low_vs_med_evoked.save(low_vs_med_evoked_file_id)  
    low_vs_med_group_comp_contrast = mc.Contrast("Low > Medium",low_vs_med_contrast.values, low_vs_med_evoked, low_vs_med_evoked.times)
    
    components_list = {}
    #high_group_comp_contrast
    #low_group_comp_contrast
    #evoked_component_contrast
    
    #HIGH VS LOW
    components_list['component'] = evoked_component
    components_list['first'] = high_group_comp_component
    components_list['second'] = low_group_comp_component        
    comp_diff = mean_high_ds.flatten() - mean_low_ds.flatten()
    f_val, p_val = stats.f_oneway(high_group, low_group)
    evoked_component.p_val = p_val[0]
    evoked_component.t_val = f_val
    evoked_component.test_name = 'high_vs_low_iq'
    
    ica_sol = ica_c.ICA(subject_tc, subject_tp, subject_tp_evoked, subject_tp_evoked.times, component=components_list)
            
    fig_component_folder = os.path.join(target_dir, 'subj', 'fig', 'component')
    fs.ensure_dir(fig_component_folder)
    component_fig_id = os.path.join(fig_component_folder, condition_name + str(comp_num) + '_component_' + str(col_num))+ '_' + str(evoked_component.test_name)
        
    title1 = "VEF Component: IC"
   
    title1 = title1 + str(col_num)
    #title = title1 + " "  + "(" + r'$p={:.5f}$'.format(p_val[0]) + "). "+ "High IQ Group > Low IQ Group."
    title = title1 + ". " + "Contrast: High IQ Group > Low IQ Group."
   
    ica_sol.plot_tc_contrast_with_topo(condition_name.upper(), col_num, fig_id = component_fig_id,title=title, scalings=1e15, annotate=True, topo_time=None)
    
    #HIGH VS MEDIUM
    components_list['component'] = evoked_component
    components_list['first'] = high_group_comp_component
    components_list['second'] = med_group_comp_component        
    comp_diff = mean_high_ds.flatten() - mean_med_ds.flatten()
    #t_val, p_val = stats.ttest_1samp(comp_diff, 0)
    f_val, p_val = stats.f_oneway(high_group, med_group)
    evoked_component.p_val = p_val[0]
    evoked_component.t_val = f_val
    evoked_component.test_name = 'high_vs_med_iq'
    
    print(p_val)
    
    ica_sol = ica_c.ICA(subject_tc, subject_tp, subject_tp_evoked, subject_tp_evoked.times, component=components_list)
            
    fig_component_folder = os.path.join(target_dir, 'subj', 'fig', 'component')
    fs.ensure_dir(fig_component_folder)
    component_fig_id = os.path.join(fig_component_folder, condition_name + str(comp_num) + '_component_' + str(col_num))+ '_' + str(evoked_component.test_name)
        
    #title = "VEF Component: IC" + str(col_num) + " "  + "(" + r'$p={:.5f}$'.format(p_val[0]) + "). "+ "High IQ Group > Medium IQ Group."
    title = "VEF Component: IC" + str(col_num) + ". "+ ". Contrast: High IQ Group > Medium IQ Group."
    ica_sol.plot_tc_contrast_with_topo(condition_name.upper(), col_num, fig_id = component_fig_id,title=title, scalings=1e15, annotate=True, topo_time=None)
    
    #LOW VS MEDIUM
    components_list['component'] = evoked_component
    components_list['first'] = low_group_comp_component
    components_list['second'] = med_group_comp_component        
    comp_diff = mean_high_ds.flatten() - mean_med_ds.flatten()
    #t_val, p_val = stats.ttest_1samp(comp_diff, 0)
    f_val, p_val = stats.f_oneway(low_group, med_group)
    evoked_component.p_val = p_val[0]
    evoked_component.t_val = f_val
    evoked_component.test_name = 'low_vs_med_iq'
    
    print(p_val)
    
    ica_sol = ica_c.ICA(subject_tc, subject_tp, subject_tp_evoked, subject_tp_evoked.times, component=components_list)
            
    fig_component_folder = os.path.join(target_dir, 'subj', 'fig', 'component')
    fs.ensure_dir(fig_component_folder)
    component_fig_id = os.path.join(fig_component_folder, condition_name + str(comp_num) + '_component_' + str(col_num))+ '_' + str(evoked_component.test_name)
        
    title = "VEF Component: IC" + str(col_num)  + "Contrast: Low IQ Group > Medium IQ Group."
    ica_sol.plot_tc_contrast_with_topo(condition_name.upper(), col_num, fig_id = component_fig_id,title=title, scalings=1e15, annotate=True, topo_time=None)
    
    #print(high_group_comp_evoked)
    #print(high_group_comp_evoked.info)
    
    #print(med_group_comp_evoked)
    #print(med_group_comp_evoked.info)
    
    #print(low_group_comp_evoked)
    #print(low_group_comp_evoked.info)
    
    print("p_val = "  + str(p_val))
    
    print(high_group_comp_evoked.data.shape)
    print(high_group_comp_evoked.info)
    
    print(subject_tp_evoked.data.shape)
    print(subject_tp_evoked.info)
    
    print("subject_tp.shape " + str(subject_tp.shape))
        
    print("mean_high_ds " + str(mean_high_ds.shape))
    
    print(subject_tp_evoked.data.shape)
    print(high_group_comp_evoked.data.shape)
    
    vef_ic = high_group_comp_evoked.plot_topomap(times=0.4, ch_type='mag', 
                                     sensors=False, colorbar=False, 
                                     res=300, size=3,
                             time_unit='ms', contours=1, image_interp='bilinear', average=0.02, axes=None, extrapolate='head')
    
    #file_fig_id = fig_id + '_component' + '_' + condition + '_' + str(col_num)
    #plt_util.save_fig_pdf_no_dpi(component_fig_id)
    #plt.close()
    
        
def generate_results_by_comp(target_dir, cond, comp_num, col_num, cols=2, figsize=(15,5), times=0.3):
    condition_name = get_condition_name(cond)
    subj_folder = os.path.join(target_dir, 'subj', 'csv')
    fig_folder = os.path.join(target_dir, 'subj', 'fig')
    fs.ensure_dir(fig_folder)
    subj_fig_id = os.path.join(target_dir, 'subj', 'csv', 'subject_bk_scores.csv')
    subjects_df = pd.read_csv(subj_fig_id)
    subject_tc = get_tc_by_path(target_dir, condition_name, comp_num)
    subject_tp = get_tp_by_path(target_dir, condition_name, comp_num)
    
    agg_ica_fig_id = os.path.join(fig_folder, condition_name + '_agg_' + str(comp_num) + '_' + str(col_num))
    agg_ica_tc_grid_fig_id = os.path.join(fig_folder, condition_name + '_tc_grid_agg_' + str(comp_num) + '_' + str(col_num))
    
    agg_ica_tc_topo_grid_fig_id = os.path.join(fig_folder, condition_name + '_tc_grid_agg_topo' + str(comp_num) + '_' + str(col_num))    
    agg_ica_tc_grid_topo_fig_id = os.path.join(fig_folder, condition_name + '_tc_topo_agg_' + str(comp_num) + '_' + str(col_num))
    
    cluster_fig_id = os.path.join(fig_folder, condition_name + '_tc_cluster_' + str(comp_num) + '_' + str(col_num))
    comp_evoked = generate_evoked(org_evoked[cond], subject_tp, cond)
    
    print(subject_tp.shape)
    
    print("Evoked = " + str(comp_evoked))
    
    print("Hi Evoked Info = " + str(comp_evoked.info))
    ica_sol = ica.ICA(subject_tc, subject_tp, comp_evoked, comp_evoked.times)
    #ica.plot_components(fig_id=fig_id)
    #ica.plot_ica_component(fig_id=fig_id)
    #ica_sol.plot_tc(condition_name.upper(),fig_id=agg_ica_fig_id)
    #ica_sol.plot_tc_grid(condition_name.upper(),fig_id = agg_ica_tc_grid_fig_id, cols=cols, figsize=figsize)
    
    ica_sol.plot_tc_with_topo(condition_name.upper(),fig_id=agg_ica_tc_grid_topo_fig_id)
    ica_sol.plot_tc_grid_with_topo(condition_name.upper(),fig_id = agg_ica_tc_grid_fig_id, cols=cols, figsize=figsize)
    #find_siginificant_times(subject_tp, 2000,comp_evoked.times, fig_id = cluster_fig_id)
    
    #mne.viz.plot_evoked_topomap
    print("Subject TP = " + str(subject_tp.shape))
    
    
    

if __name__ == "__main__":
    
    generate_results_by_component(ica_results_vis2['4'], '4', 2, 1, 1, figsize=(15, 3))
    generate_results_by_component(ica_results_vis3['4'], '4', 3, 2, 1, figsize=(15, 3))
  
    generate_results_by_component(ica_results_vis5['4'], '4', 5, 1, 1, figsize=(15, 3))
    generate_results_by_component(ica_results_vis5['4'], '4', 5, 2, 1, figsize=(15, 3))
    
    generate_results_by_component(ica_results_vis20['4'], '4', 20, 16, 1, figsize=(15, 3))
    generate_results_by_component(ica_results_vis20['4'], '4', 20, 18, 1, figsize=(15, 3))
    
    generate_results_by_component(ica_results_vis25['4'], '4', 25, 4, 1, figsize=(15, 3))
    
    #generate_results_by_comp(ica_results_vis2['4'], '4', 2, 2, 2, figsize=(15, 3), times=0.3)
    #generate_results_by_comp(ica_results_vis3['4'], '4', 3, 3, 2)
    #generate_results_by_comp(ica_results_vis5['4'], '4', 5, 5, figsize=(15, 7))
    #generate_results_by_comp(ica_results_vis10['4'], '4', 10, 10, figsize=(15, 15))
    #generate_results_by_comp(ica_results_vis15['4'], '4', 15, 15, figsize=(15, 15))
    #generate_results_by_comp(ica_results_vis20['4'], '4', 20, 20, figsize=(15, 20))
    #generate_results_by_comp(ica_results_vis25['4'], '4', 25, 25, figsize=(15, 25))

