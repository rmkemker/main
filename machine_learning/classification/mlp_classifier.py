"""
Name: mlp_classifier.py
Author: Ronald Kemker
Description: Classification pipeline for multi-layer perceptron

Note:
Requires keras and and scikit-learn
https://keras.io/
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Nadam
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2

from machine_learning.parallelizer import Parallelizer
from machine_learning.model_checkpoint_parallel import ModelCheckpoint as MC2
from machine_learning.preprocessing import ImagePreprocessor
from machine_learning.utils import train_test_split_per_class as split_folds


class MLPClassifier(BaseEstimator, ClassifierMixin):
  
    def __init__(self, hidden_layer_shape = [64], weight_decay=1e-4,
                 batch_normalization=True, activation='relu', save_fname=None,
                 patience = 6, lr=2e-3, min_lr = 2e-6, verbose = 2, mu=None,
                 refit = False, gpu_list = None, optimizer=None, nb_epochs=1000,
                 kernel_initializer = 'glorot_normal', lr_patience = 3):
        
        self.model = Sequential()
        self.hidden = hidden_layer_shape
        self.wd = weight_decay
        self.bn = batch_normalization
        self.activation = activation
        self.fname = save_fname
        self.patience = patience
        self.lr = lr
        self.min_lr = min_lr
        self.verbose = verbose
        self.mu = mu
        self.epochs = nb_epochs
        self.refit = refit
        self.gpus = gpu_list
        self.ki = kernel_initializer
        self.lr_patience = lr_patience
                                       
        if optimizer is None:
            self.opt = Nadam(self.lr)
            
        if self.refit:
            raise NotImplementedError('I have not implemented the refit functionality yet.')
    
    def _model(self, input_shape):

        self.model.add(Dense(self.hidden[0], 
                        input_shape=(input_shape[1],), 
                        kernel_regularizer=l2(self.wd),
                        kernel_initializer=self.ki))
        if self.bn:
            self.model.add(BatchNormalization(axis=1))
        self.model.add(Activation(self.activation))
        
        for i in self.hidden[1:]:
            self.model.add(Dense(i, kernel_regularizer=l2(self.wd),
                                 kernel_initializer=self.ki))
            if self.bn:
                self.model.add(BatchNormalization(axis=1))
            self.model.add(Activation(self.activation))

        
        self.model.add(Dense(self.N, activation='softmax',
                             kernel_regularizer=l2(self.wd),
                             kernel_initializer=self.ki))
    
    def fit(self, X_train, y_train, X_val, y_val, batch_size=256):

        self.N = len(np.bincount(y_train))

        X_train = np.float32(X_train)
        X_val = np.float32(X_val)
        
        self._model(X_train.shape)            
        if self.gpus is not None:
            par = Parallelizer(self.gpus)
            self.model = par.transform(self.model)
            batch_size *= len(self.gpus)

        cb = []
        cb.append(EarlyStopping(monitor='val_loss', patience=self.patience, 
                           verbose=self.verbose, mode='auto'))
        
        if self.fname is not None:
            
            if self.gpus is None:           
                cb.append(ModelCheckpoint(self.fname, monitor='val_loss', 
                                     verbose=self.verbose, 
                                     save_best_only=True, mode='auto'))
                
            else:
                cb.append(MC2(self.fname, monitor='val_loss', 
                                     verbose=self.verbose, 
                                     save_best_only=True, mode='auto'))                
        
        cb.append(ReduceLROnPlateau(patience=self.lr_patience, 
                               min_lr = self.min_lr, verbose=self.verbose))


        if self.mu is None:
            hist = np.ones(self.N)
        else:
            hist = self.mu*np.log(len(y_train)/np.bincount(y_train))
            hist[hist<1] = 1
        
        wght = {}
        for i in range(self.N):
            wght[i] = hist[i]
            

        if self.N > 2:
            self.model.compile(optimizer=self.opt, metrics=['accuracy'], 
                               loss='categorical_crossentropy')
            y_train = np.float32(to_categorical(y_train, self.N))
            y_val = np.float32(to_categorical(y_val, self.N))
            
        else:
            self.model.compile(optimizer=self.opt, metrics=['accuracy'], 
                               loss='binary_crossentropy')

        history = self.model.fit(X_train, y_train,
            epochs=self.epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val, y_val),
            callbacks=cb, class_weight=wght, verbose=self.verbose) 

        
        if self.fname is not None:
            self.model = load_model(self.fname)   
            #if refit:
                #TODO: Implement this
        return history
    
    def predict(self,X):
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
#if __name__ == "__main__":
#    from keras.datasets import mnist
#    from machine_learning.metrics import Metrics
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#    x_train = x_train.reshape(60000, -1)
#    x_test = x_test.reshape(10000, -1)
#    
#    
#    mlp = MLPClassifier(hidden_layer_shape=[64], gpu_list=[0,1,2], save_fname='tmp.hdf5')
#    mlp.fit(x_train, y_train, x_test, y_test, 128)
#    
#    pred = mlp.predict(x_test)
#    M = Metrics(y_test, pred)
#    print('Mean-Class Accuracy = %3.2f%%' % (M.mean_class_accuracy() * 100.0))