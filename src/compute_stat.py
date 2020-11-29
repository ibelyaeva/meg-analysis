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


org_subjects = {}
org_subjects['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map.csv'
org_subjects['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map.csv'
org_subjects['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map.csv'
org_subjects_meta = '/pl/meg/analysis/ica_output/data/subject_map_with_scores.csv'

org_subjects_index = {}
org_subjects_index['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map_index.csv'
org_subjects_index['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map_index.csv'
org_subjects_index['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map_index.csv'

bk_subject_path = {}
bk_subject_path['6'] = '/pl/meg/analysis/ica_output/data/session1/jica/aud/reconstructed'
bk_subject_path['4'] = '/pl/meg/analysis/ica_output/data/session1/jica/vis/reconstructed' 
bk_subject_path['2'] = '/pl/meg/analysis/ica_output/data/session1/jica/aud_vis/reconstructed'    

ica_results_aud = {} 
ica_results_aud['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5'
ica_results_aud['4'] = '/pl/meg/analysis/ica_output/processed/gfp/vis/5'
ica_results_aud['2'] = '/pl/meg/analysis/ica_output/processed/gfp/aud_vis/5'

cond_name = {}
cond_name['6'] = 'aud'
cond_name['4'] = 'vis'
cond_name['2'] = 'aud_vis'

subj_results5 = {} 
subj_results5['6'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'
subj_results5['4'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'
subj_results5['2'] = '/pl/meg/analysis/ica_output/processed_gfp/aud/5/subj'

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

clustered_data = '/pl/meg/analysis/ica_output/data/subject_iq_scores_ses_clustered.csv'
cog_group_path = '/pl/meg/analysis/ica_output/data/all_sessions_demo_beh_scores_processed.csv'
org_subjects_scores_path = '/pl/meg/analysis/ica_output/data/subject_map_with_scores.csv'

import pandasql as ps
import scipy
from scipy.stats import ttest_ind

def get_condition_name(cond):
    return cond_name[cond]

def get_matlab_solution_path(target_dir, condition_name, comp_num):
    subject_group_name =  condition_name  + '_' + str(comp_num) + '_ica_br1' + '.mat' 
    print("Target Directory =" + str(target_dir))
    subject_group_path = os.path.join(target_dir, subject_group_name)
    return subject_group_path

def get_tc(map_name, comp, cond):
    
    condition_name = get_condition_name(cond)
    ica_path = map_name[cond]
    file_name = 'tc'+ '_' + condition_name + '_' + str(comp) + '.npy'
    tc_path = os.path.join(ica_path, 'np', file_name)
    tc = np.load(tc_path)
    return tc

def get_tp(map_name, comp, cond):
    
    condition_name = get_condition_name(cond)
    ica_path = map_name[cond]
    file_name = 'tp'+ '_' + condition_name + '_' + str(comp) + '.npy'
    tp_path = os.path.join(ica_path, 'np', file_name)
    tp = np.load(tp_path)
    return tp

def get_subject__data(map_name):
    subject_data = pd.read_csv(map_name)
    
    return subject_data    

def get_subject_res_folder(cond):
    
    csv_folder = os.path.join(subj_results5['6'], 'csv')
    fs.ensure_dir(csv_folder)
    return csv_folder

def generate_with_cog_group(subject_path, cog_path):
    subject_df = pd.read_csv(subject_path)
    subject_df_with_scores =  subject_df
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
        #group_map[s[1]] = s[9]
        print(s[1])
        print(s[9])
        print(s[10])
        
    for s in subject_list:
        row = {}
        row['subject_name'] = s[0]
        row['subject_path'] = s[1]
        row['fsiq'] = s[2]
        row['cognition_composite_score'] = s[3]
        row['age'] = s[4]
        row['gender'] = s[5]
        row['math_lab_file_path'] = s[6]
        row['fif_gfp_file_path'] = s[7]
        try:
            row['cluster_id'] = score_map[s[0]]
            row['cog_group'] = group_map[s[0]]
        except:
            print("Not found Cluster Id")
        
        subjects.append(row)
    
    
    subjects_map_df = pd.DataFrame(subjects)
    print(subjects_map_df)
        
    subjects_map_df.to_csv(org_subjects_scores_path)
    print("Saved Subjects with Cognition Scores @" + org_subjects_scores_path)
    return subjects_map_df
       
def get_tc_by_path(target_dir, condition_name, comp_num):
    tc_path = os.path.join(target_dir, 'subj/data/agg/' + 'tc_' +condition_name + '_' + str(comp_num) + '.npy')
    tc = np.load(tc_path)
    print("Reading TC @ " + tc_path + "; TC.shape: " + str(tc.shape))
    return tc
    
def create_df(t):
    df = pd.DataFrame(t)
    return df
    
def add_index(df):
    cnt  = 1
    row = []
    for k in range(len(df)):
        row.append(cnt)
        cnt = cnt + 1
    df['k'] = row
    return df
        

def gender_t_test(tc, cond, comp_num):
    
    gender_tc = tc.T    
    
    gender_tc = scipy.stats.zscore(gender_tc)   
    col_names = ['subject_name', 'subject_path', 'fsiq', 'cognition_composite_score', 'age', 'gender']
    subjects = pd.read_csv(org_subjects_index['6'], sep=',', usecols=col_names)
 
    cog_df = pd.read_csv(cog_group_path, sep=',')
    
    #subjects.set_index('subject_name').join(cog_df.set_index('subject_name'), on='subject_name', how='left')
    #subjects_index_df = subjects.merge(cog_df, how='left', left_on='subject_name', right_on='subject_name') 
    
    subject_fig_id = os.path.join(get_subject_res_folder(cond) , 'subject_index.csv')
  
    print ("Subjects DF = " + str(len(subjects)))
    
    row = []
    cnt = 1
    for i in range (len(subjects)):
        row.append(cnt)
        cnt = cnt + 1
    
    subjects['k'] = row
    subjects.to_csv(subject_fig_id)
    
    print("Saved Subject DF @" + subject_fig_id)
    print("Saved Subject DF " + str(subjects))
      
    row = []
    tc_df = pd.DataFrame(data=gender_tc, columns=["tc1", "tc2","tc3", "tc4", "tc5"])    
    cnt = 1
    for i in range (len(tc_df)):
        row.append(cnt)
        cnt = cnt + 1
    
    tc_df['k'] = row
    
    tc_fig_id = os.path.join(get_subject_res_folder(cond) , 'tc_gender.csv')  
    tc_df.set_index('k').join(subjects.set_index('k'), on='k', how='left')
    
    tc_df = tc_df.merge(subjects, how='left', left_on='k', right_on='k') 
    print ("TC DF = " + str(len(tc_df)))    
    tc_df.to_csv(tc_fig_id, index=False)
   
    t, p = ttest_ind(*tc_df.groupby('gender')['tc2'].apply(lambda x:list(x)))
    
    #t, p = ttest_ind(*tc_df.groupby('fsiq')['tc1'].apply(lambda x:list(x)))
    
    print(t,p)
    
    subjects_index_df = subjects.merge(cog_df, how='left', left_on='subject_name', right_on='subject_name')
    subjects_index_fig_id = os.path.join(get_subject_res_folder(cond) , 'subject_cog_index.csv')
    
    row = []
    cnt = 1
    
    for i in range (len(subjects_index_df)):
        row.append(cnt)
        cnt = cnt + 1
    
    subjects_index_df['k'] = row
    subjects_index_df.to_csv(subjects_index_fig_id)
    
    print("Saved Subject DF @" + subjects_index_fig_id)
    print("Saved Subject COG DF " + str(subjects_index_df))
    
    tc_df_cog = tc_df.merge(subjects_index_df, how='left', left_on='subject_name', right_on='subject_name') 
    print("TC COG DF " + str(tc_df_cog.head()))
    tc_cog_fig_id = os.path.join(get_subject_res_folder(cond) , 'tc_cog.csv') 
    tc_df_cog.to_csv(tc_cog_fig_id, index=False) 
   
    high_group = ps.sqldf("select tc5 from tc_df_cog where cluster_id = 3")
    print("TC HIGH COG DF " + str(len(high_group)))
    
    med_group = ps.sqldf("select tc5 from tc_df_cog where cluster_id = 2")
    print("TC MED COG DF " + str(len(med_group)))
    
    low_group = ps.sqldf("select tc5 from tc_df_cog where cluster_id = 1")
    print("TC LOW COG DF " + str(len(low_group)))
    
    f, p = scipy.stats.f_oneway(high_group, med_group, low_group)
    
    print(f,p)
   
    compute_fsat_cog_score(tc_df_cog, comp_num, "tc1", 1, get_subject_res_folder(cond))
    compute_fsat_cog_score(tc_df_cog, comp_num, "tc2", 2, get_subject_res_folder(cond))
    compute_fsat_cog_score(tc_df_cog, comp_num, "tc3", 3, get_subject_res_folder(cond))
    compute_fsat_cog_score(tc_df_cog, comp_num, "tc4", 4, get_subject_res_folder(cond))
    compute_fsat_cog_score(tc_df_cog, comp_num, "tc5", 5, get_subject_res_folder(cond))
 
def compute_fsat_cog_score(tc_df_cog, comp_num, tc_num, num, output_dir):   
    
    df = tc_df_cog
    
    sql = "select cog_group,"  +  tc_num + " as value from df"
    data_df = ps.sqldf(sql)
    
    # Ordinary Least Squares (OLS) model
    model = ols('value ~ C(cog_group)', data=data_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    print(anova_table)
    anova_table_name = 'anova_'+ str(comp_num) + '_' + str(num)+ '.csv' 
    
    fig_id_anova = os.path.join(output_dir, anova_table_name)  
    anova_table.to_csv(fig_id_anova) 
    
    print("Saving anova_table table @ " + str(fig_id_anova))
    
    m_comp = pairwise_tukey(data=data_df, dv='value', between='cog_group')
    
    m_comp_name = 'm_comp_ptukey_'+ str(comp_num) + '_' + str(num)+ '.csv' 
    fig_id_p_tukey = os.path.join(output_dir, m_comp_name)
    m_comp.to_csv(fig_id_p_tukey)
    print(m_comp)
    
    pvals = m_comp['p-tukey'].tolist()
    print(pvals)

    reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
    print(reject, pvals_corr)
    
    rows = []
    row = {}
    row['reject'] = reject
    row['pvals'] = pvals_corr
    
    rows.append(row)
    
    p_tukey_data = pd.DataFrame(rows)
    print(p_tukey_data)
    
    ptukey_name = 'ptukey_'+ str(comp_num) + '_' + str(num)+ '.csv' 
    fig_id_p_tukey = os.path.join(output_dir, ptukey_name)  
    p_tukey_data.to_csv(fig_id_p_tukey) 
    
def make_df(col1, col2):
    df = pd.DataFrame()
    df['tc'] = col1
    df['group'] = col2
    return df
    
def run_test_by_column(col1, col2, test_name, output_dir, comp_num, col_num):
    data_df = make_df(col1, col2)

    # Ordinary Least Squares (OLS) model
    model = ols('tc ~ C(group)', data=data_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    print(anova_table)
    
    anova_table_name = test_name + '_anova_'+ str(comp_num) + '_colnum_' + str(col_num)+ '.csv' 
    
    fig_id_anova = os.path.join(output_dir, anova_table_name)  
    anova_table.to_csv(fig_id_anova) 
    
    print("Saving anova_table table @ " + str(fig_id_anova))
    
    m_comp = pairwise_tukey(data=data_df, dv='tc', between='group')
    m_comp_name = 'm_comp_ptukey_'+ str(comp_num) + '_colnum_' + str(col_num)+ '.csv' 
    
    fig_id_p_tukey = os.path.join(output_dir, m_comp_name)
    m_comp.to_csv(fig_id_p_tukey)
    print(m_comp)
    pvals = m_comp['p-tukey'].tolist()
    print(pvals)

    reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
    print(reject, pvals_corr)
    
    rows = []
    row = {}
    row['reject'] = reject
    row['pvals'] = pvals_corr
    
    rows.append(row)
    
    p_tukey_data = pd.DataFrame(rows)
    print(p_tukey_data)
    
    ptukey_name = 'ptukey_'+ str(comp_num) + '_colnum_' + str(col_num)+ '.csv' 
    fig_id_p_tukey = os.path.join(output_dir, ptukey_name)  
    p_tukey_data.to_csv(fig_id_p_tukey) 
    
    data_df['value'] = data_df['group']
    fig_id_group_df = os.path.join(output_dir, test_name + '_column_group_summary' + '_' + str(col_num)) +'.csv' 
    data_df.sort_values(by='group', ascending=False, inplace=True)
    data_df.to_csv(fig_id_group_df) 
    print("Saving summary group stat @ " + str(fig_id_group_df))
    
    
    return reject, pvals_corr
    
def run_t_tests(target_dir, condition_name, comp_num):  
    subject_matlab_path = get_matlab_solution_path(target_dir, condition_name, comp_num)
    print("Subject Matlab Path = " + subject_matlab_path)
    subject_meta_df = generate_with_cog_group(org_subjects['6'], cog_group_path)
    tc = get_tc_by_path(target_dir, condition_name, comp_num)

    cnt = 1
    result_dir = os.path.join(target_dir, 'subj/csv/stat') 
    fs.ensure_dir(result_dir)
    print("Running T-tests for = " + target_dir) 
    
    gend_sign = []
    print ("Gender T-TEST")
    for col in tc:
        
        zscored_col = scipy.stats.zscore(col) 
        print("Current Column Cnt = " + str(cnt) + "; Colum Length = " + str(len(zscored_col)))
        #gender t-test
        print("Len = " + str(len(list(subject_meta_df['gender']))))
        reject, pvals_corr = run_test_by_column(zscored_col, list(subject_meta_df['gender']), 'gender', result_dir,comp_num, cnt)
        cnt = cnt + 1
        
        if reject.any():
            row = {}
            row['col'] = cnt
            row['comp'] = comp_num
            gend_sign.append(row)
            
    cog_sign = []
    print ("Cognition Score T-TEST")
    cnt = 1    
    for col in tc:
        zscored_col = scipy.stats.zscore(col) 
        print("Current Column Cnt = " + str(cnt) + "; Colum Length = " + str(len(col)))
        #gender t-test
        print("Len = " + str(len(list(subject_meta_df['gender']))))
        reject, pvals_corr = run_test_by_column(zscored_col, list(subject_meta_df['cog_group']), 'cog_group', result_dir,comp_num, cnt)
        cnt = cnt + 1
        
        if reject.any():
            row = {}
            row['col'] = cnt
            row['comp'] = comp_num
            cog_sign.append(row)
            
    gend_sign_df = pd.DataFrame(gend_sign)
    cog_sign_df = pd.DataFrame(cog_sign)
        
    gen_sign_id = os.path.join(result_dir, condition_name + str(comp_num) + '_gen_sign_id.csv')
    gend_sign_df.to_csv(gen_sign_id)
    print("Saved Gender Significance @" + gen_sign_id)
    
    cog_sign_id = os.path.join(result_dir, condition_name + str(comp_num) + '_cog_sign_id.csv')
    cog_sign_df.to_csv(gen_sign_id)
    print("Saved Cognition Significance @" + str(comp_num) + cog_sign_id)
   
def run_t_ind_comp_test(map_name, comp, cond):
    tc = get_tc(map_name, comp, cond)
    print ("Loaded TC = " + str(tc.shape))
    
    tp = get_tp(map_name, comp, cond)
    print ("Loaded TP = " + str(tp.shape))
    gender_t_test(tc, cond, comp)

def run_aud_t_tests():
    condition_name = get_condition_name('6')
    print("Condition Name =" + condition_name )
    run_t_tests(ica_results_aud5['6'], condition_name, 5)
    run_t_tests(ica_results_aud10['6'], condition_name, 10)
    run_t_tests(ica_results_aud15['6'], condition_name, 15)
    run_t_tests(ica_results_aud20['6'], condition_name, 20)
    run_t_tests(ica_results_aud25['6'], condition_name, 25)
    
def run_vis_t_tests():
    condition_name = get_condition_name('4')
    print("Condition Name =" + condition_name )
                       
    run_t_tests(ica_results_vis2['4'], condition_name, 2)
    run_t_tests(ica_results_vis3['4'], condition_name, 3)
    run_t_tests(ica_results_vis5['4'], condition_name, 5)
    run_t_tests(ica_results_vis10['4'], condition_name, 10)
    run_t_tests(ica_results_vis15['4'], condition_name, 15)
    run_t_tests(ica_results_vis20['4'], condition_name, 20)
    run_t_tests(ica_results_vis25['4'], condition_name, 25)

if __name__ == "__main__":
    #run_aud_t_tests()
    run_vis_t_tests()

    
    