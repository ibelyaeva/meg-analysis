import copy
from nilearn import plotting
import os
import numpy as np
import io_util as io

tsc_file_pattern = 'tsc_cost_'
tsc_z_cost_pattern  = 'tsc_z_cost_'
rse_cost_pattern = 'rse_cost_'
solution_cost_pattern = 'solution_cost_'
train_cost_pattern = 'train_cost_'
file_extension = ".csv"

def get_parent_name(file_path):
    current_dir_name = os.path.split(os.path.dirname(file_path))[1]
    return current_dir_name

def get_subjects(root_dir):
    subject_list = []
    for root,d_names,f_names in os.walk(root_dir):
        for f in f_names:
            if f.startswith("swa"):
                file_path = os.path.join(root, f)
                subject_list.append(file_path)
    return subject_list

def get_subjects_with_prefix(root_dir, prefix):
    subject_list = []
    for root,d_names,f_names in os.walk(root_dir):
        for f in f_names:
            if f.startswith(prefix):
                file_path = os.path.join(root, f)
                subject_list.append(file_path)
    return subject_list

def traverse_dir_with_prefix(root_dir, prefix):
    tsc_list = []
    for root,d_names,f_names in os.walk(root_dir):
        for f in f_names:
            if f.startswith(prefix):
                file_path = os.path.join(root, f)
                tsc_list.append(file_path)
    return tsc_list


