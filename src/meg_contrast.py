import numpy as np
import os as os
import pandas as pd
import file_service as fs
import csv
import mne
from pathlib import Path

from statsmodels.formula.api import ols
from pingouin import pairwise_tukey
import statsmodels.api as sm
import pingouin as pg

import scipy

import pandasql as ps
from scipy import stats


class Contrast(object):
    def __init__(self, name, data, evoked, times, short_name=None, pval=None):
        self.contrast_name = name
        self.data = data 
        self.info = evoked.info
        self.evoked = evoked
        self.times = times
        self.p_val = pval
        
        if short_name is None:
            self.short_name = name
            
        
        
    def compute_stat(self):
        compare_with = np.zeros_like(self.data)
        t, p = stats.ttest_1samp(self.data,compare_with)
        return t, p
        pass