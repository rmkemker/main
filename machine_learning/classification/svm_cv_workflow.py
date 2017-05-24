"""
Name: svm_cv_workflow.py
Author: Ronald Kemker
Description: Classification pipeline for support vector machine which uses 
             k-fold cross-validation
Note:
Requires scikit-learn
"""

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class SVM_Workflow():
    
    def __init__(self, kernel='linear', standard_scaler=True, n_jobs=8,
                 verbosity = 10, pca_components=None, whiten=True, cv=3, 
                 max_iter=1000):
        self.kernel = kernel
        self.scale = standard_scaler
        self.n_jobs = n_jobs
        self.verbose = verbosity
        self.pca = pca_components
        self.whiten = whiten
        self.cv = cv
        self.max_iter = max_iter
                
    def fit(self, train_data, train_labels):
        
        if self.scale:
            self.sclr = StandardScaler()
            train_data = self.sclr.fit_transform(train_data)
        
        if self.pca is not None:
            self.pca = PCA(n_components=self.pca, whiten=self.whiten)
            train_data = self.pca.fit_transform(train_data)
            
            if self.verbose:
                print('Data reduced to %d features.' % train_data.shape[-1])
        
        if self.kernel == 'linear':        
            clf = LinearSVC(class_weight='balanced', dual=False,random_state=6,
                      multi_class='ovr',max_iter=self.max_iter)
        
            #Cross-validate over these parameters
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float)}
        elif self.kernel == 'rbf':
            clf = SVC(random_state=6, class_weight='balanced', cache_size=8000,
                      decision_function_shape='ovr',max_iter=self.max_iter, 
                      tol=1e-4)            
            params = {'C': 2.0**np.arange(-9,16,2,dtype=np.float),
                      'gamma': 2.0**np.arange(-15,4,2,dtype=np.float)}

        #Coarse search      
        gs = GridSearchCV(clf, params, refit=False, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=self.cv)
        gs.fit(train_data, train_labels)
        
        #Fine-Tune Search
        if self.kernel == 'linear':
            best_C = np.log2(gs.best_params_['C'])
            params = {'C': 2.0**np.linspace(best_C-2,best_C+2,10,
                                            dtype=np.float)}
        elif self.kernel == 'rbf':
            best_C = np.log2(gs.best_params_['C'])
            best_G = np.log2(gs.best_params_['gamma'])
            params = {'C': 2.0**np.linspace(best_C-2,best_C+2,10,
                                            dtype=np.float),
                      'gamma': 2.0**np.linspace(best_G-2,best_G+2,10,
                                                dtype=np.float)}            
        
        self.gs = GridSearchCV(clf, params, refit=True, n_jobs=self.n_jobs,  
                          verbose=self.verbose, cv=self.cv)
        return self.gs.fit(train_data, train_labels)
                
    def predict(self, test_data):
        
        if self.scale:
            test_data = self.sclr.transform(test_data)
        
        if self.pca is not None:
            test_data = self.pca.transform(test_data)
        
        return self.gs.predict(test_data)
        
    def predict_proba(self, test_data):
        if self.scale:
            test_data = self.sclr.transform(test_data)
        
        if self.pca is not None:
            test_data = self.pca.transform(test_data)
        
        self.gs.predict_proba(test_data)        
