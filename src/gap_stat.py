import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs

import pandas as pd
import matplotlib.pyplot as plt
import random
from numpy import zeros

mrn_demog_path = '/pl/meg/metadata/csv/nm_dev_cog_demographics.csv'
mrn_scores_path = '/pl/meg/metadata/csv/nm_beh_scores.csv'
session1_path = '/pl/meg/data/meg_ms/MRN/session1'
session2_path = '/pl/meg/data/meg_ms/MRN/session2'
session3_path = '/pl/meg/data/meg_ms/MRN/session3'
session23_path = '/pl/meg/data/meg_ms/MRN/session23'
all_sessions_nm_path = '/pl/meg/data/meg_ms/MRN/subjects_nm.csv'
all_sessions_folder = '/pl/meg/data/meg_ms/MRN'
iq_score_path = '/pl/meg/data/meg_ms/MRN/subject_iq_scores.csv'
iq_score_clustered_path = '/pl/meg/data/meg_ms/MRN/subject_iq_scores_clustered.csv'

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])
    
def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

def gap_statistic(X):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,10)
    Wks = zeros(len(ks))
    Wkbs = zeros(len(ks))
    sk = zeros(len(ks))
    for indk, k in enumerate(ks):
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

def run():
    file_id = os.path.join(all_sessions_folder, 'beh_scores.csv')
    beh_data = pd.read_csv(file_id)
    beh_data.drop('cognition_composite_score', axis=1, inplace=True)
    beh_data.dropna(axis=0, inplace=True)
    beh_data.to_csv(iq_score_path)
    
    beh_data_df = pd.DataFrame() 
    beh_data_df['fsiq'] = beh_data['fsiq']
    X = beh_data_df.values

if __name__ == '__main__':
    run()