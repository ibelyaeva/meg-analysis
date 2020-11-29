import kmeans1d

import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

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
iq_score_ses_clustered_path = '/pl/meg/data/meg_ms/MRN/subject_iq_scores_ses_clustered.csv'
all_sessions_demo_beh_scores_path = '/pl/meg/data/meg_ms/MRN/all_sessions_demo_beh_scores_raw.csv'
all_sessions_processed_path = '/pl/meg/data/meg_ms/MRN/all_sessions_demo_beh_scores_processed.csv'

def get_clusters():
    file_id = os.path.join(all_sessions_folder, 'beh_scores.csv')
    beh_data = pd.read_csv(file_id)
    beh_data.drop('cognition_composite_score', axis=1, inplace=True)
    beh_data.dropna(axis=0, inplace=True)
    beh_data.to_csv(iq_score_path)
    continuous_features = ['fsiq']
   
    file_demo_id = os.path.join(all_sessions_folder, mrn_demog_path)
    demo_data = pd.read_csv(file_demo_id)
    print(demo_data)
    
    beh_data_clusterred = beh_data
    beh_data_clusterred.drop('Unnamed: 0', axis=1, inplace=True)
    beh_data_df = pd.DataFrame() 
    beh_data_df['fsiq'] = beh_data['fsiq']
    
    print(beh_data_df)
    print(beh_data_df[continuous_features].describe())
    
    mms = MinMaxScaler(feature_range=(0, 1), copy=True)
    mms.fit(beh_data_df)
    data_transformed = mms.transform(beh_data_df)
    
    sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data_transformed)
        sum_of_squared_distances.append(km.inertia_)
        
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k for Subject IQ Scores')
    plt.close()
    #plt.show()
    
    colors = ['red','green','blue']
    x = beh_data_df['fsiq'].tolist()
    clusters, centroids = kmeans1d.cluster(x, 3)
    print(clusters)   
    print(centroids) 
    
    
    clusters_new = []
    for c in clusters:
        clusters_new.append(c+1)
    
    print(len(clusters))
    print(clusters_new)
    beh_data_clusterred['cluster_id'] = clusters_new
    beh_data_clusterred.to_csv(iq_score_clustered_path)
    
    beh_data_clust = beh_data_clusterred.copy(True)
    
    beh_data_clust.set_index('subject').join(demo_data.set_index('URSI'), on='subject', how='left')
    
    beh_data_clust = beh_data_clust.merge(demo_data, how='left', left_on='subject', right_on='URSI')
    beh_data_clust.drop("URSI", axis=1, inplace=True)
    beh_data_clust.drop("Age", axis=1, inplace=True)
    beh_data_clust.drop("Gender", axis=1, inplace=True)
    values = {'SES': 50}
    beh_data_clust = beh_data_clust.fillna(value=values)
    
    beh_data_clust.to_csv(iq_score_ses_clustered_path)
    
    print("beh_data_clust")
    print(beh_data_clust)
    print ("===")
     
    for c in clusters_new:
        print ("Cluster = " + str(c))
    
    print ("===")
    
    for c in clusters_new:
        tempdf = beh_data_clusterred[beh_data_clusterred['cluster_id'] == c]
        tempdf_path = '/pl/meg/data/meg_ms/MRN/results/iq_cluster_stat' + '_' + str(c) + '.csv'
        tempdf.to_csv(tempdf_path)
        
    #print(beh_data_clusterred)
    
    cluster_df = pd.DataFrame()
    cluster_df['cluster_id']=beh_data_clusterred['cluster_id']
    cluster_df['fsiq']=beh_data_clusterred['fsiq']   
    
    
    cluster_df_values = cluster_df.values
    
    mds = MDS(2,random_state=0)
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    cluster_scaled =scaler.fit_transform(cluster_df_values)
    cluster_2d = mds.fit_transform(cluster_scaled)
    
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)
    
    labels =['low', 'medium', 'high']
    
    for i in np.unique(cluster_df['cluster_id']):
        subset = cluster_2d[cluster_df['cluster_id'] == i]
  
        label = labels[i-1]
        
        x = [row[0] for row in subset]
        y = [row[1] for row in subset]
        plt.scatter(x,y,c=colors[i-1],label=label)
    
    plt.legend(loc=(0.1, 0.8))
    plt.ylabel('Normalized Min-max IQ')
    plt.title('Total Subjects = 109. Subjects Clustered by IQ Score')
    plt.close()
    #plt.show()
    
    
    pass

def compute_gap_statistics():
    file_id = os.path.join(all_sessions_folder, 'beh_scores.csv')
    beh_data = pd.read_csv(file_id)
    beh_data.drop('cognition_composite_score', axis=1, inplace=True)
    beh_data.dropna(axis=0, inplace=True)
    beh_data.to_csv(iq_score_path)
    
    beh_data_df = pd.DataFrame() 
    beh_data_df['fsiq'] = beh_data['fsiq']
    
    k, gapdf = optimalK(beh_data_df.values, nrefs=1, maxClusters=8)
    
    iq_gap_stat_path = '/pl/meg/data/meg_ms/MRN/results/iq_gap_stat.csv'
    gapdf.to_csv(iq_gap_stat_path)
    
    print(gapdf)
    print(k)
    
    
def compute_clusters():
    file_id = os.path.join(all_sessions_folder, 'beh_scores.csv')
    beh_data = pd.read_csv(file_id)
    beh_data.drop('cognition_composite_score', axis=1, inplace=True)
    beh_data.dropna(axis=0, inplace=True)
    beh_data.to_csv(iq_score_path)
    continuous_features = ['fsiq']
    #print(beh_data)
    
    beh_data_df = pd.DataFrame() 
    beh_data_df['fsiq'] = beh_data['fsiq']
    print(beh_data_df[continuous_features].describe())
    
    x = beh_data_df['fsiq'].tolist()
    clusters, centroids = kmeans1d.cluster(x, 3)
    
    cluster_df = pd.DataFrame()
    cluster_path_path = '/pl/meg/data/meg_ms/MRN/results/iq_cluster_stat.csv'
    cluster_df['cluster_id'] = clusters
    cluster_df.to_csv(cluster_path_path)
    
    centroids_df = pd.DataFrame()
    centroids_path_path = '/pl/meg/data/meg_ms/MRN/results/iq_centroid_stat.csv'
    centroids_df['centroid'] = centroids
    centroids_df.to_csv(centroids_path_path)   
    
    print(clusters)   
    print(centroids) 
    
    
    
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    
def postprocess():
    raw_beh_data = pd.read_csv(all_sessions_demo_beh_scores_path)
    raw_beh_data.drop(raw_beh_data.columns[[0]], axis=1, inplace=True)
    
    iq_clustered_data = pd.read_csv(iq_score_clustered_path)
    iq_clustered_data.drop(iq_clustered_data.columns[[0]], axis=1, inplace=True)
    
    all_sessions_processed = raw_beh_data.copy(True)
    all_sessions_processed = all_sessions_processed.merge(iq_clustered_data, how='left', left_on='subject_name', right_on='subject')
    all_sessions_processed.drop("subject", axis=1, inplace=True)
    all_sessions_processed.drop("fsiq_y", axis=1, inplace=True)
    all_sessions_processed.rename(columns={"fsiq_x": "fsiq"}, inplace=True)
    
    cog_group = {1: "Low",  2: "Medium", 3: "High"}
    cog_group_df = pd.DataFrame( {"cog_group_map": [1, 2, 3]} )
    all_sessions_processed['cog_group'] = all_sessions_processed["cluster_id"].map(cog_group)
    all_sessions_processed.to_csv(all_sessions_processed_path)
    

    pass

if __name__ == '__main__':
    #get_clusters()
    #compute_gap_statistics()
    #compute_clusters()
    postprocess()