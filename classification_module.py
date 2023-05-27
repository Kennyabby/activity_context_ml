#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from  sklearn.ensemble import RandomForestClassifier
import pickle


# In[21]:


# The Model Class is used to perform train set balancing as well as model classifications and fitting
class Model:
    def __init__(self,features, target, models={'RandomForestClassifier':RandomForestClassifier, 
                                                'KNeighborsClassifier': KNeighborsClassifier, 
                                                'LogisticRegression':LogisticRegression, 
                                                'MLPClassifier': MLPClassifier}):
        
        self.models = models
        self.features = features
        self.target = target        
    
    # Splits the features and target into their train and test sets
    def split(self, test_size=0.2, random_state=0):
        features = self.features
        target = self.target
        return train_test_split(features, target, test_size=test_size, random_state=random_state)
    
    # Initializes the creates models
    def initialize_models(self, n_neighbors=7, n_estimators=150):
        models = list(self.models.values())
        dict_keys = list(self.models.keys())
        self.dict_keys = dict_keys
        md_list = []
        print('Creating models...')
        for i,model in enumerate(models):
            if i==0:                
                md = model(n_estimators=n_estimators)
            elif i==1:
                md = model(n_neighbors=n_neighbors)
            elif i==3:
                md = model(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)
            else:
                md = model()
            md_list+=[md]
        self.init_models = md_list
        print('Models created successfully!')
    
    # Balances the Imbalanced classes.
    def split_and_balance (self, test_size=0.2, random_state=0):
        
        print('splitting train-test data...')
        X_train, X_test, y_train, y_test = self.split(test_size, random_state)
        print('Splitted Successfully!')
        
        print('Balancing train data set...')
        
        # Perfoming class balancing using resampling (oversampling)
        X_train_balanced = np.empty((0, X_train.shape[1]))
        y_train_balanced = np.empty(0)
        
        #Performing oversampling for each class
        for class_label in np.unique(y_train):
            X_class = X_train[y_train == class_label]
            y_class = y_train[y_train == class_label]
            X_class_balanced, y_class_balanced = resample(X_class, y_class, replace=True, n_samples = X_train[y_train == 0].shape[0],
                                                         random_state = 42)

            X_train_balanced = np.concatenate((X_train_balanced, X_class_balanced), axis =0)
            y_train_balanced = np.concatenate((y_train_balanced, y_class_balanced), axis =0)
        print('Balanced Successfully!')
        self.X_train = X_train_balanced
        self.y_train = y_train_balanced
        self.y_test = y_test
        self.X_test = X_test
    
    # Fits the models
    def fit_models(self, n_neighbors=7, n_estimators=150, balance = True):
        self.initialize_models(n_neighbors, n_estimators)
        self.split_and_balance()
        
        print('Fitting models...')
        for i,md in enumerate(self.init_models):
            print(f'Fitting {self.dict_keys[i]} Model ...')
            md.fit(self.X_train, self.y_train)
        print('All models fitted Successfully!')
        return self.init_models
    
    # Predicts the y_test of the X_test testing set.
    def predict_test (self):
        fitted_models = self.init_models
        y_preds = []
        for fit in fitted_models:
            y_preds += [fit.predict(self.X_test)]
        self.y_preds = y_preds


# In[26]:


# Computes performance metric scores of the models
class PerformanceMetric:
    
    def __init__ (self, y_preds, y_test):        
        self.y_preds = y_preds
        self.y_test = y_test
        
    # Computes the accuracy of the models
    def get_accuracy (self):
        pred_accuracy = []
        for y_pred in self.y_preds:
            pred_accuracy += [accuracy_score(self.y_test, y_pred)]
        self.pred_accuracy = pred_accuracy
        return self.pred_accuracy
    
    # Computes the precision of the models
    def get_precision (self):
        pred_precision = []
        for y_pred in self.y_preds:
            pred_precision += [precision_score(self.y_test, y_pred, average='weighted')]
        self.pred_precision = pred_precision
        return self.pred_precision
    
    # Computes the recall of the models
    def get_recall (self):
        pred_recall = []
        for y_pred in self.y_preds:
            pred_recall += [recall_score(self.y_test, y_pred, average='weighted')]
        self.pred_recall = pred_recall
        return self.pred_recall
    
    # Computes the f1_score of the models
    def get_f1_score (self):
        pred_f1_score = []
        for y_pred in self.y_preds:
            pred_f1_score += [f1_score(self.y_test, y_pred, average='weighted')]
        self.pred_f1_score = pred_f1_score
        return self.pred_f1_score
    
    # Calculates the Performance metrics scores above
    def get_performance (self):
        accuracy = self.get_accuracy()
        precision = self.get_precision()
        recall = self.get_recall()
        f1_score = self.get_f1_score()
        
        performance_list = []
        for i in range(len(self.y_preds)):
            per_dict = {
                'accuracy': accuracy[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1_score[i]
            }
            performance_list+=[per_dict]
        
        return performance_list
 


# In[ ]:




