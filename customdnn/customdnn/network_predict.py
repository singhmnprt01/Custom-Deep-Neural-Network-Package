#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: singhmnprt01@gmail.com

customdnn Copyright (C) 2020 singhmnprt01@gmail.com
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from customdnn.network_train import TrainingDeepNetwork

class PredictDeepNetwork:
    
    NETWORK_CONSTANT = 2
    
    def __inti__(self):
        self.param = {}
           
    def predict_proba(self, x_test,y_test,param,network_size=3):
        self.param = param
        d = TrainingDeepNetwork()
        nw_size=network_size + self.NETWORK_CONSTANT
        Z_test_All, A_test_all = {},{}
        Z_test_all, A_test_all = self.__forward_prop_test(param,x_test,y_test,nw_size,d)
        y_test_hat = A_test_all['A'+str(nw_size-1)]        
        return y_test_hat   
    
    def __forward_prop_test(self,param,x,y,size_nn,d):
        A = x
        A_prev = A   
        A_all ={}
        Z_all = {}    
    
        for i in range (1,size_nn-1):
            W = param['W'+str(i)]
            A_prev = A
            b = param['b' + str(i)]
            
            Z = np.dot(W,A_prev) + b 
            A = d.relu(Z)
            
            Z_all['Z' + str(i)] = Z
            A_all['A' + str(i)] = A    
            
        ## calculate output layer Z & A using Sigmoid ##   
        W = param['W'+str(size_nn - 1)]
        A_prev = A_all['A' + str(size_nn-2)]
        b = param['b' + str(size_nn - 1)]
        
        Z= np.dot(W,A_prev) + b
    
        A = d.sigmoid(Z)
        
        Z_all['Z' + str(size_nn-1)] = Z
        A_all['A'+str(size_nn-1)] = A
        
        return (Z_all, A_all)    
    
    def nn_auc(self, y_true,y_pred):
         auc_nn = round(roc_auc_score(np.squeeze(y_true),np.squeeze(y_pred)),3)
         return auc_nn
     
        
