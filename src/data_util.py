import os


folder_path_subject1 = "/pl/mlsp/data/cobre/cobreRST/control/M87100994"
data_path_subject1   = "swaAMAYER+cobre01_63001+M87100944+20110309at135133+RSTpre_V01_R01+CM.nii"
data_path_subject2   = "swaAMAYER+cobre01_63001+M87103074+20091208at183208+RSTpre_V01_R01+CM.nii"
data_path_subject3   = "swaAMAYER+cobre01_63001+M87103074+20091208at183208+RSTpre_V01_R01+CM.nii"

ellipsoid_masks_folder = "/work/pl/sch/analysis/results/masked_images/ellipsoid_masks"
ellipsoid_mask1_path = "size_20_7_15.nii"
ellipsoid_mask2_path = "size_20_11_25.nii"
ellipsoid_mask3_path = "size_35_20_25.nii"

def get_parent_name(file_path):
    current_dir_name = os.path.split(os.path.dirname(file_path))[1]
    return current_dir_name

def get_subjects(root_dir):
    subject_list = []
    for root,d_names,f_names in os.walk(root_dir):
        for f in f_names:
            if f.endswith("fif"):
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


def traverse(dirname):
    
    subject_list = []
    for dirpath, dirs, files in os.walk(dirname):    
        folder_path = os.path.dirname(dirpath)
        print ("Dir Path: " + str(folder_path)) 
        subject_path = get_subject(folder_path)
        if subject_path is not None:
            print ("Subject path = " + str(subject_path))
            subject_list.append(subject_path)
    return subject_list

def get_subject(file_path):
    
    subject_path = None
    for filename in os.listdir(file_path):    
        folder_path = file_path
        print ("Folder Path: " + str(folder_path)) 
        fname = os.path.join(folder_path,filename)
        print ("Current Path: " + str(fname))
        print ("File Name: " + str(filename)) 
        if filename.startswith("fif"):
                subject_path = fname
                print ("Adding subject: " + subject_path)
                return subject_path
                break
        else:
                print("skip =" + fname)
    return subject_path