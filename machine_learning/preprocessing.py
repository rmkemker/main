"""
Name: preprocessing.py
Author: Ronald Kemker
Description: Image Pre-processing workflow
Note:
Requires scikit-learn
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

class ZCAWhiten():
    
    def __init__(self, epsilon=0.1):
        self.eps = epsilon
        
    def fit(self, X):       
        self.sclr = StandardScaler(with_std=False)
        X = self.sclr.fit_transform(X)
                
        X = X.T
        sigma = np.dot(X, X.T)/X.shape[-1] 
        U,S,V = np.linalg.svd(sigma) 
        self.ZCAMatrix = np.dot(np.dot(U, 
                            1.0/np.sqrt(np.diag(S) + self.eps)), U.T)                    
                
    def transform(self, X):
        X = self.sclr.transform(X)
        X = X.T
        return np.dot(self.ZCAMatrix, X).T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class GlobalContrastNormalization():
    
    def __init__(self, sqrt_bias = 10, epsilon = 1e-8, with_std = True, scale=1):
        self.sqrt_bias = sqrt_bias
        self.eps= epsilon
        self.scale = scale
        self.with_std = with_std
        
    def fit(self, X):
        
        assert X.ndim == 2, "X.ndim must be 2"
        self.scale = float(self.scale)
        assert self.scale >= self.eps
        
        self.mean = X.mean(axis=1)
        X = X - self.mean[:, np.newaxis]
        
        if self.with_std:
            ddof = 1
            if X.shape[1] == 1:
                ddof = 0
            self.normalizers = np.sqrt(self.sqrt_bias + X.var(axis=1, ddof=ddof)) / self.scale
        else:
            self.normalizers = np.sqrt(self.sqrt_bias + (X ** 2).sum(axis=1)) / self.scale
        
        self.normalizers[self.normalizers < self.eps] = 1.
        
    def transform(self, X):
        X = X - self.mean[:, np.newaxis]
        return X/self.normalizers[:, np.newaxis]
    
    def fit_transform(self , X):
        self.fit(X)
        return self.transform(X)
        

class ImagePreprocessor():
    
    def __init__(self, mode='StandardScaler' , feature_range=[0,1], 
                 with_std=True, PCA_components=None, svd_solver='auto'):
        self.mode = mode
        self.with_std = with_std
        if mode == 'StandardScaler':
            self.sclr = StandardScaler(with_std=with_std)
        elif mode == 'MinMaxScaler':
            self.sclr = MinMaxScaler(feature_range = feature_range)
        elif mode == 'PCA':
            self.sclr = PCA(n_components=PCA_components,whiten=True, 
                            svd_solver=svd_solver)
        elif mode == 'GlobalContrastNormalization':
            self.sclr = GlobalContrastNormalization(with_std=with_std)
        elif mode == 'ZCAWhiten':
            self.sclr = ZCAWhiten()
            
    
    def fit(self, data):
        data = data.reshape(-1,data.shape[-1])
        self.sclr.fit(data)
    def transform(self,data):
        sh = data.shape
        data = data.reshape(-1,sh[-1])
        data = self.sclr.transform(data)
        return data.reshape(sh)
    def fit_transform(self, data):
        sh = data.shape
        data = data.reshape(-1,sh[-1])
        data = self.sclr.fit_transform(data)
        return data.reshape(sh)
    def get_params(self):
        if self.mode == 'StandardScaler':
            if self.with_std:
                return self.sclr.mean_ , self.sclr.scale_
            else:
                return self.sclr.mean_
        elif self.mode == 'MinMaxScaler':
            return self.sclr.data_min_, self.sclr.data_max_
        elif self.mode == 'PCA':
            return self.sclr.components_
        elif self.mode == 'GlobalContrastNormalization':
            return self.sclr.mean, self.sclr.normalizers
        elif self.mode == 'ZCAWhiten':
            return self.sclr.ZCAMatrix
           
if __name__ == "__main__":     
    sclr = ImagePreprocessor('GlobalContrastNormalization')
    x = np.random.rand(100, 32,32, 3) * 255
    sclr.fit(x)
    f = sclr.transform(x)
    f2 = sclr.fit_transform(x)
    mean, normalizer = sclr.get_params()
