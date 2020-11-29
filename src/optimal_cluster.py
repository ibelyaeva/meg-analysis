import kmeans1d

import numpy as np
import csv
import pandas as pd
import os
import data_util as du
import file_service as fs

import pandas as pd
import matplotlib.pyplot as plt
import random
from k_plus_plus import KPlusPlus
from numpy import zeros

class DetK(KPlusPlus):
    def fK(self, thisk, Skm1=0):
        X = self.X
        Nd = len(X[0])
        a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
        self.find_centers(thisk, method='++')
        mu, clusters = self.mu, self.clusters
        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk  
 
    def _bounding_box(self):
        X = self.X
        xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
        ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
        return (xmin,xmax), (ymin,ymax)        
         
    def gap(self, thisk):
        X = self.X
        (xmin,xmax), (ymin,ymax) = self._bounding_box()
        self.init_centers(thisk)
        self.find_centers(thisk, method='++')
        mu, clusters = self.mu, self.clusters
        Wk = np.log(sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
                    for i in range(thisk) for c in clusters[i]]))
        # Create B reference datasets
        B = 10
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax), \
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            kb = DetK(thisk, X=Xb)
            kb.init_centers(thisk)
            kb.find_centers(thisk, method='++')
            ms, cs = kb.mu, kb.clusters
            BWkbs[i] = np.log(sum([np.linalg.norm(ms[j]-c)**2/(2*len(c)) \
                              for j in range(thisk) for c in cs[j]]))
        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2)/float(B))*np.sqrt(1+1/B)
        return Wk, Wkb, sk
     
    def run(self, maxk, which='both'):
        ks = range(1,maxk)
        fs = zeros(len(ks))
        Wks,Wkbs,sks = zeros(len(ks)+1),zeros(len(ks)+1),zeros(len(ks)+1)
        # Special case K=1
        self.init_centers(1)
        if which == 'f':
            fs[0], Sk = self.fK(1)
        elif which == 'gap':
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        else:
            fs[0], Sk = self.fK(1)
            Wks[0], Wkbs[0], sks[0] = self.gap(1)
        # Rest of Ks
        for k in ks[1:]:
            self.init_centers(k)
            if which == 'f':
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
            elif which == 'gap':
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
            else:
                fs[k-1], Sk = self.fK(k, Skm1=Sk)
                Wks[k-1], Wkbs[k-1], sks[k-1] = self.gap(k)
        if which == 'f':
            self.fs = fs
        elif which == 'gap':
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
        else:
            self.fs = fs
            G = []
            for i in range(len(ks)):
                G.append((Wkbs-Wks)[i] - ((Wkbs-Wks)[i+1]-sks[i+1]))
            self.G = np.array(G)
     
    def plot_all(self):
        X = self.X
        ks = range(1, len(self.fs)+1)
        fig = plt.figure(figsize=(18,5))
        # Plot 1
        ax1 = fig.add_subplot(131)
        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)
        ax1.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
        tit1 = 'N=%s' % (str(len(X)))
        ax1.set_title(tit1, fontsize=16)
        # Plot 2
        ax2 = fig.add_subplot(132)
        ax2.set_ylim(0, 1.25)
        ax2.plot(ks, self.fs, 'ro-', alpha=0.6)
        ax2.set_xlabel('Number of clusters K', fontsize=16)
        ax2.set_ylabel('f(K)', fontsize=16) 
        foundfK = np.where(self.fs == min(self.fs))[0][0] + 1
        tit2 = 'f(K) finds %s clusters' % (foundfK)
        ax2.set_title(tit2, fontsize=16)
        # Plot 3
        ax3 = fig.add_subplot(133)
        ax3.bar(ks, self.G, alpha=0.5, color='g', align='center')
        ax3.set_xlabel('Number of clusters K', fontsize=16)
        ax3.set_ylabel('Gap', fontsize=16)
        foundG = np.where(self.G > 0)[0][0] + 1
        tit3 = 'Gap statistic finds %s clusters' % (foundG)
        ax3.set_title(tit3, fontsize=16)
        ax3.xaxis.set_ticks(range(1,len(ks)+1))
        plt.savefig('detK_N%s.png' % (str(len(X))), \
                     bbox_inches='tight', dpi=100)