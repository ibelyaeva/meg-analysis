import numpy as np
import data_util as du
import pandas as pd
import metric_util as mt
import mri_draw_utils as mrd
import os
import shutil
import file_service as fs


def list_subjects(root, save_dir, prefix, report_name):
    
    cnt = 0
    subject_list = du.get_subjects_with_prefix(root, prefix) 
    cnt = len(subject_list)   
    df = pd.DataFrame(subject_list)
    file_name = report_name
    fig_id = os.path.join(save_dir, file_name)
    df.to_csv(fig_id)
    print ("Directory: " + str(root) + 
           "; Subject Count with Prefix " + prefix + "; " + str(cnt))


def copy_to(root_dir, target_dir, search_prefix, prefix, suffix=None):
    subject_list = du.get_subjects_with_prefix(root_dir, search_prefix)
    
    for subject in subject_list:
        subject_name = du.get_parent_name(subject)
        
        new_folder_name = prefix + "_" + subject_name
        
        target_folder = os.path.join(target_dir, new_folder_name)
        fs.ensure_dir(target_folder)
        
        new_file_name = prefix + "_" + subject_name
        if suffix is not None:
            new_file_name = new_file_name + '_' + suffix + '.nii'
        else:
            new_file_name = new_file_name + '.nii'
            
        file_path = os.path.join(target_folder, new_file_name)
        print ("Copy " + subject + '->' + file_path)
        shutil.copy(subject, file_path)


if __name__ == "__main__":
    
    subjects_root_dir = "/pl/mlsp/data/cobre/subjects/"
    report_dir = "/work/fnc/subject_list"
    report_name = 'subjects.csv'
    list_subjects(subjects_root_dir, report_dir, 'swa', report_name)
    
    data_dir = "/pl/mlsp/data/cobre/subjects/ica_analysis"
    #copy_to(subjects_root_dir, data_dir, 'swa', 'org', suffix='x_true')
    report_name = 'subjects_original_data.csv'
    list_subjects(data_dir, data_dir, 'org', report_name)
    
    subjects_root_dir_x_hat = "/work/scratch/tensor_completion/4D/multi_subject/meta_subject2020-08-24_05_05_13/random"
    
    report_name = 'subjects_rs-scg-tt.csv'
    list_subjects(subjects_root_dir_x_hat, data_dir, 'x_hat', report_name)
    #copy_to(subjects_root_dir, data_dir, 'swa', 'rs_scg_tt', suffix='x_hat')
    
    list_subjects(data_dir, data_dir, 'rs_scg_tt', report_name)
    
    subjects_root_dir_x_hat_cp = "/work/scratch/tensor_completion/4D/cp/multisubject/meta_subject2020-08-25_02_00_02/random"
    
    report_name = 'subjects_cp.csv'
    list_subjects(subjects_root_dir_x_hat_cp, data_dir, 'x_hat', report_name)
    
    #copy_to(subjects_root_dir_x_hat_cp, data_dir, 'x_hat', 'cp', suffix='x_hat')
    report_name = 'subjects_cp.csv'
    list_subjects(data_dir, data_dir, 'cp', report_name)

    subjects_root_dir_x_hat_tucker = "/work/scratch/tensor_completion/4D/tucker/multisubject/meta_subject2020-08-25_00_51_13/random"
    report_name = 'subjects_tucker.csv'
    list_subjects(subjects_root_dir_x_hat_tucker, data_dir, 'x_hat', report_name)
    
    #copy_to(subjects_root_dir_x_hat_tucker, data_dir, 'x_hat', 'tucker', suffix='x_hat')
    list_subjects(data_dir, data_dir, 'tucker', report_name)
    
    subjects_root_dir_x_hat = "/work/scratch/tensor_completion/4D/multi_subject/meta_subject2020-08-24_05_05_13/random"
    report_name = 'subjects_x_miss.csv'
    list_subjects(subjects_root_dir_x_hat, data_dir, 'x_miss', report_name)
    copy_to(subjects_root_dir_x_hat, data_dir, 'x_miss', 'x_miss', suffix='x_miss')
    list_subjects(subjects_root_dir_x_hat, data_dir, 'x_miss', report_name)
