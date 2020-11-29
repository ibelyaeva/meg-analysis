import nilearn
import texfig
import os
import matplotlib.pyplot as plt

'''
Saves plots at the desired location 
'''
        
def save_fig_png(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".png")
        print("Saving figure", path)
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_fig_pdf(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".pdf")
        print("Saving figure", path)
        plt.savefig(path, format='pdf', dpi=300)
        plt.close()
        
def save_fig_pdf_no_dpi(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".pdf")
        print("Saving figure", path)
        plt.savefig(path, format='pdf', tight_layout=tight_layout)
        plt.close()


def save_fig_abs_path(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", path)
    plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300, bbox_inches='tight')
    

def save_csv_by_path(df, file_path, dataset_id):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)

def save_csv_by_path_adv(df, file_path, dataset_id, index = False):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path, index = index)