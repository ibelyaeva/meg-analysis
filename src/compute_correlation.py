import numpy as np

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
import hoggorm as ho

ica_results_aud5 = {} 
ica_results_aud5['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5'
ica_results_aud5['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/5'
ica_results_aud5['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/5'

ica_results_aud10 = {} 
ica_results_aud10['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/10'
ica_results_aud10['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/10'
ica_results_aud10['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/10'

ica_results_aud15 = {} 
ica_results_aud15['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/15'
ica_results_aud15['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/15'
ica_results_aud15['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/15'

ica_results_aud20 = {} 
ica_results_aud20['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/20'
ica_results_aud20['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/20'
ica_results_aud20['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/20'


ica_results_aud25 = {} 
ica_results_aud25['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/25'
ica_results_aud25['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/25'
ica_results_aud25['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/25'


cond_name = {}
cond_name['6'] = 'aud'
cond_name['4'] = 'vis'
cond_name['2'] = 'aud_vis'

def get_condition_name(cond):
    return cond_name[cond]

def get_corr_coeff(x):
    x_stand = ho.standardise(x, mode=0)
    corr_coeff = ho.RVcoeff([x_stand, x_stand])
    return corr_coeff

def get_corr(folder_path, cond, comp_num):
    condition_name = get_condition_name(cond)
    ica_path = os.path.join(folder_path, condition_name + '_' + str(comp_num) + '_' + 'ica.mat')
    ica_res = convert_to_mat.mat_numpy(ica_path)
    
    bk_mat_name = condition_name  + '_' + str(comp_num) + '_ica_br1' + '.mat' 
    bk_mat_path = os.path.join(folder_path, bk_mat_name)
    bk = convert_to_mat.mat_numpy(bk_mat_path)
    bk_set = bk['compSet']
    tc_bk = bk_set['timecourse'].item()
    tp_bk = bk_set['topography'].item()
    
    c = np.mean(get_corr_coeff(tc_bk.T))
    print ("Cond #" + condition_name + "; Component #:" + str(comp_num) + "; Corr. Coeff = " + str(c))
    return c

if __name__ == "__main__":
    
    aud_corr = []
    row = {}
    
    row[5] = get_corr(ica_results_aud5['6'], '6', 5)
    row[10] = get_corr(ica_results_aud10['6'], '6', 10)
    row[15] = get_corr(ica_results_aud15['6'], '6', 15)
    row[20] = get_corr(ica_results_aud20['6'], '6', 20)
    row[25] = get_corr(ica_results_aud25['6'], '6', 25)
    
    
        