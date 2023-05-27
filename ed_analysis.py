#!/usr/bin/env python
# coding: utf-8

# In[108]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from  sklearn.ensemble import RandomForestClassifier
import pickle


# In[109]:


# reads data from file path specified in the parameter.
def read_data(path):
#    initialized error as false
    error = False
    try:
#       read file from path
        pd_data = pd.read_csv(path)
        
    except IOError as err:
        print('Could not find file path',path+'.')
        # set error to be true
        error = True
    finally:
        # if error is false return pd_data
        if not error:
            return pd_data


# In[110]:


# performs data cleaning. "data" must be a pandas type parameter
def clean_data(data, drop_list=[]):
    
    drop_data = data
    
    # drops the features in the drop_list parameter.
    if len(drop_list)!= 0:        
        drop_data = data.drop(drop_list, axis = 1)
        
    # This drops the all rows that contain Null values.
    cleaned_data = drop_data.dropna()
    
    return cleaned_data


# In[111]:


# Splits data into its target and features
class DataFeaturizer:
    # "data" type: pandas object
    def __init__ (self, data):
        self.data = data
        self.features = []
        self.processed_features = []        
        self.target = []
    
    '''set_features: extracts the features and target of the data (pandas type) provided.
        parameters: (column (target column): 'string', encode_target (LabelEncoder): 'boolean', 
        feature_processor(sklearn.preprocessing):'string' ['standard_scalar', 'normalizer'] )
    '''
    def set_features (self, column, encode_target=False, feature_processor=''):
        feature_processors = ['standard_scalar', 'normalizer', '']
        target = self.data[column]
        features = self.data.drop(column, axis=1)
        if feature_processor == 'standard_scalar':
            scaler = StandardScaler()
            self.processed_features = scaler.fit_transform(features)
        elif feature_processor == 'normalizer':
            normalizer = Normalizer()
            self.processed_features = normalizer.fit_transform(features)
        elif feature_processor not in feature_processors:
            print('class DataFeaturizer does not have the processor requested. Pick from the following list:',feature_processors[0:2])
        self.features = features
        
        if encode_target:
            encoder = LabelEncoder()
            self.target = encoder.fit_transform(target)
        else:
            self.target = target
    
    '''get_features: returns features as a pandas type variable.'''
    def get_features(self):
        features = self.features
        if len(features)!=0:
            return features
        else:
            print('feature is currently an empty list. Initialize feature with set_features method.')
            return []
    '''get_processed_features: returns procces_features as an array type variable.'''
    def get_processed_features(self):
        processed_features = self.processed_features
        if len(processed_features)!=0:
            return processed_features
        else:
            print('processed_feature is currently an empty list. Pass feature_processor parameter in the set_features method.')
            return []
    '''get_target: returns target as a pandas type variable.'''
    def get_target(self):
        target = self.target
        if len(target)!=0:
            return self.target
        else:
            print('target is currently an empty list. Initialize target with set_features method.')
            return []


# In[114]:


# Extracts statistical features from raw features
class FeatureExtractor:
    def __init__ (self, feature_array):
        self.feature_array = feature_array        
        self.stats_dict = {}        
        self.feature_matrix = []
    
    # Extracts mean feature
    def set_mean(self):
        mean = np.mean(self.feature_array, axis = 1, keepdims=True)
        self.mean = mean
    def get_mean(self):
        return self.mean
    
    # Extracts median feature
    def set_median(self):
        median = np.median(self.feature_array, axis = 1, keepdims=True)
        self.median = median
    def get_median(self):
        return self.median
    
    # Extracts variance feature
    def set_variance(self):
        variance = np.var(self.feature_array, axis = 1, keepdims = True)
        self.variance = variance
    def get_variance(self):
        return self.variance
    
    # Extracts standard deviation feature
    def set_std_dev(self):
        std_dev = np.std(self.feature_array, axis = 1, keepdims = True)
        self.std_dev = std_dev
    def get_std_dev(self):
        return self.std_dev
    
    # Extracts sum of squres (sos) feature
    def set_sos(self):
        sos = np.sum((self.feature_array - self.mean) ** 2, axis = 1, keepdims = True)
        self.sos = sos
    def get_sos(self):
        return self.sos
    
    # Extracts root mean square (rms) feature
    def set_rms(self):
        rms = np.sqrt(np.mean(self.feature_array ** 2, axis = 1, keepdims=True))
        self.rms = rms
    def get_rms(self):
        return self.rms
    
    # Extracts zero-crossing feature
    def set_zero_crossing(self):
        zero_crossing = (np.sum(np.abs(np.diff(np.sign(self.feature_array))), axis = 1, keepdims=True)/self.feature_array.shape[1])
        self.zero_crossing = zero_crossing
    def get_zero_crossing(self):
        return self.zero_crossing
    
    # Initializes the statistical features extraction
    def set_stats(self, pre=''):
        self.set_mean()
        self.set_median()
        self.set_variance()
        self.set_std_dev()
        self.set_sos()
        self.set_rms()
        self.set_zero_crossing()
        
        features_label = ['mean', 'median', 'std_dev', 'variance', 'sos', 'rms', 'zero_crossing']
        extracted_features = [self.mean, self.median, self.std_dev, self.variance, self.sos, self.rms, self.zero_crossing]
        stats_dict ={}
        for i in range(len(features_label)):
            stats_dict[pre+features_label[i]] = extracted_features[i]
        self.stats_dict = stats_dict
    def get_stats(self):
        return self.stats_dict
    
    # Combines all features into an array
    def set_feature_matrix(self):
        feature_matrix = np.concatenate(tuple(list(self.stats_dict.values())), axis = 1)
        self.feature_matrix = feature_matrix
    def get_feature_matrix(self):
        self.set_feature_matrix()
        return self.feature_matrix
    
    # Coverts the feature_matrix into a Dataframe
    def get_feature_df(self):
        self.set_feature_matrix()
        feature_df = pd.DataFrame(self.feature_matrix, columns = list(self.stats_dict.keys()))
        return feature_df
    
    


# In[113]:


# Selects best features that is suitable for model input.
class FeatureSelector:
    def __init__ (self, feature_matrix, target, stats_dict={}):
        self.feature_matrix = feature_matrix
        self.target = target
        self.stats_dict = stats_dict
        self.mi_dict = {}
    
    # Computes mutual information scores
    def set_mi_scores(self):
        mi_scores = mutual_info_classif(self.feature_matrix, self.target)
        mi_dict = {}
        for i, score in enumerate(mi_scores):
            mi_dict[list(self.stats_dict.keys())[i]] = f"{score:.3f}" 
        self.mi_dict = mi_dict
    def get_mi_scores(self):
        self.set_mi_scores()
        return self.mi_dict
    
    # Selects Best Features using mutual_info_classif
    def get_selected_features(self, k):
        # Selects top k features with highest mutual information
        selector = SelectKBest(mutual_info_classif, k=k)
        new_features = selector.fit_transform(self.feature_matrix, self.target)
        
        # Gets the indices of the selected features
        selected_indices = selector.get_support(indices=True)
        
        # Creates a new table with only the selected features
        selected_features = []
        for i in selected_indices:
            selected_features += [list(self.stats_dict.keys())[i]]
        return selected_features


# In[ ]:




