import numpy as np
import io_functions as io
import file_service as fs
import data_util as du
import mne
from scipy import stats
import os
import matplotlib.pyplot as plt

cond_name = {}
cond_name['6'] = 'aud'
cond_name['4'] = 'vis'
cond_name['2'] = 'aud_vis'


grand_avg_cond = {} 
grand_avg_cond['6'] = '/pl/meg/analysis/ica_output/processed/aud/grand/'
grand_avg_cond['4'] = '/pl/meg/analysis/ica_output/processed/aud/grand/'
grand_avg_cond['2'] = '/pl/meg/analysis/ica_output/processed/aud_vis/grand/'


grand_avg = {} 
grand_avg['6'] = '/pl/meg/analysis/ica_output/processed/aud/figures/grand/'
grand_avg['4'] = '/pl/meg/analysis/ica_output/processed/aud/figures/grand/'
grand_avg['2'] = '/pl/meg/analysis/ica_output/processed/aud_vis/figures/grand/'

ica_results = {} 
ica_results['6'] = '/pl/meg/analysis/ica_output/data/session1/aud/reconstructed/ica'
ica_results['4'] = '/pl/meg/analysis/ica_output/data/session1/vis/reconstructed/ica'
ica_results['2'] = '/pl/meg/analysis/ica_output/data/session1/aud_vis/reconstructed/ica'

def plot_grand_average(source_dir, condition):
    target_dir = grand_avg_cond[condition]
    fs.ensure_dir(target_dir)
    
    subject_list = du.get_subjects(source_dir) 
    cnt = len(subject_list)  
    print ("Subject Count: " + str(cnt))
    
    all_evokeds = [list() for _ in range(0)]
    
    grand_averages = [] 
    
    colours = ['white', 'blue', 'green', 'purple', 'yellow',
               'red', 'orange', 'pink',
               'grey']
    
    for s in subject_list:
        file_name = s
        evoked = io.read_evokeds_by_path_and_channel_type(s, 'mag',baseline = (-100, 0), condition=condition, verbose=True)
        grand_averages.append(evoked) 
        
    
    combine_evoked = mne.grand_average(grand_averages)     
    grand_avg_file_id = os.path.join(grand_avg_cond[condition], cond_name[condition] + '_' + 'grand_ave.fif')
    print("Writing Grand Average for Condition =" + str(cond_name[condition]) + "; " + grand_avg_file_id)
    mne.write_evokeds(grand_avg_file_id, combine_evoked)
    
    target_dir = os.path.join(grand_avg[condition], cond_name[condition])
    fs.ensure_dir(target_dir)
    grand_avg_plot_fig_id = os.path.join(target_dir + '_' + 'grand_ave.pdf') 
    
    print("Saving figure @ " + str(grand_avg_plot_fig_id))
    
    combine_evoked.plot_topomap();
    plt.savefig(grand_avg_plot_fig_id)
    plt.close()
    

if __name__ == '__main__':
    
    aud_ica = ica_results['6']
    plot_grand_average(aud_ica, '6')
    
    vis_ica = ica_results['4']
    plot_grand_average(vis_ica, '4')
      
    aud_vis_ica = ica_results['2']
    plot_grand_average(aud_vis_ica, '2')
    pass