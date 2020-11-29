import numpy as np
import io_functions as io
import file_service as fs
import data_util as du
import mne
from scipy import stats
import os
import matplotlib.pyplot as plt

org_subjects = {}
org_subjects['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/subjects/subject_map.csv'
org_subjects['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/subjects/subject_map.csv'
org_subjects['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/subjects/subject_map.csv'

event_dict = {'auditory': '6', 'visual': '4', 'aud_visual': '2'}

def plot_events(condition):
    
    fif_raw_file_path = '/work/temp/M87160162/M87160162_multisensory_session1_tsss_mc_proj_raw-eve.fif'
    file_path = '/work/temp/M87160162/M87160162_eyesclosedopen_session1_tsss_mc-eve.fif'
    #raw = mne.io.read_raw_fif(fif_raw_file_path, verbose=True)
    
    events = mne.read_events(file_path, verbose=True)
    
    #fig = mne.viz.plot_events(events, sfreq=1000,
    #                      first_samp=0, show=True)
    #fig.subplots_adjust(right=0.7)
    #plt.tight_layout()
    #plt.show()
   
    pass

if __name__ == '__main__':
    plot_events('6')