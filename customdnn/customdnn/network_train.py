#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: singhmnprt01@gmail.com

customdnn Copyright (C) 2020 singhmnprt01@gmail.com
"""

### 4th End to End DNN Code using NumPy only ####

### This code has additional features & Hyperparameters of DNN namely:-
# Dropout
# Normalizing/Scaling Inputs
# Initializaing weights with better condition
# Adam, gdm, rmsprop
# Mini- Batch

# Train-Test Split
# Check model performance using AUC (Area under the curve)
# Generalised the network - user input to create layers number and neurons per layer in the network
# Load file from your storage (csv or excel) and run the DNN on it !
# ---- Full fledge customized DNN class & Package
# epoch numbers from user
# adam, gdm, rmsprop choice as user input

# check cost_array calculation incase of minibatches !!!
# ---- Implement batch-norm
# clip the final output value between .9999 to .0001
# ---- early stopping !
# default parameter initialization
# change gradient form int to string input

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split

"""
The entire code is desined using NumPy.

It takes user inputs as dataset (x and y) and other network designing inputs namely:-
learning_rate ---- The rate of learning at which the gradient steps will be taken to minimize the cost
beta1         ---- Beta constant for Gradient Descent Momemtum Optimization Algorithm
beta2         ---- Beta constant for Root Mean Square prop Optmization Algorithm
batch_size    ---- To create custmized mini-batches to amplify the processing and improve model accuracy/generalization/learning.
network_size  ---- A custom variable to design the number of layer of your network. It is exclusive of input and output layer
gradient      ---- Gradient Descent Optimization algorithm choosing field. You can input any of the following three :-
    #### GDM        - Gradient Descent Momentum
    #### RMSprop    - Room Mean Square Prop
    #### Adam       - Adaptive Momentum Estimation
epoch_num     ---- Number of epochs/iterations for the network. 

"""
class SplitData:
    """
     This classhelps the user to split the data into train and test.
     User needs to input x, y and the Test percentage
     
    """
    def split_train_test(self,x,y,test_percentage):
        
        """
        x - The feature dataset as a DATAFRAME
        y - Target variable as a DATAFRAME
        test_percentage - percetntage of total data to be used as test. Enter an integer value.
        (if test_percentage is 20, it says that 20% data will test and 80% willl be train)
        
        The fiunction returns 4 values (matrices):-
        x_train,x_test,y_train,y_test
        
        """
        
        test_ratio = test_percentage/100
        
        x= x.to_numpy()
        y= y.to_numpy()
        
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        ##### Train - Test Split
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_ratio, stratify = y)
        
        ### Transposing again to make it fit for the NN design
        x_train = x_train.T
        x_test  = x_test.T
        y_train = y_train.T
        y_test  = y_test.T
        return x_train,x_test,y_train,y_test
    

class TrainingDeepNetwork: 
     
    def __init__(self):
        self.layer_nn=[]
    
    def param_init(self,layers_nn):
        
        layer_nn = self.layer_nn
        np.random.seed(1)
        param={}
        size_nn = len(layer_nn)
        for i in range(1,size_nn):
            param['W' + str(i)] = np.random.random((layer_nn[i],layer_nn[i-1])) * np.sqrt(2/layer_nn[i-1]) ## initializing appropriate weights 
            param['b' + str(i)] = np.zeros((layer_nn[i],1))
            
            ### Checker to check the dimensions of weights w & bias b ###
            assert(param['W'+str(i)].shape == (layer_nn[i], layer_nn[i-1]))
            assert(param['b'+str(i)].shape == (layer_nn[i],1))
            
        return param
      
    def var_init(self,layer_nn):
        
 
        cost_array = []
        
        param={}
        param = self.param_init(layer_nn)
            
        Z_all, A_all = {},{}
        dW_all,dZ_all,Vdw, Vdb,Sdw,Sdb = {},{},{},{},{},{}
        return cost_array, param, Z_all, A_all, dZ_all, dW_all,Vdw,Vdb,Sdw,Sdb    
      
    def dnn_preprocessing(self,x,y,batch_size,layer_nn):
        
        layer_nn = self.layer_nn
        mini_batches =[]

        ## Variable intialization
        cost_array, param, Z_all, A_all, dZ_all, dW_all,Vdw,Vdb,Sdw,Sdb  = self.var_init(layer_nn)
        
        ### Mini-Btches creation
        mini_batches = self.create_mini_batch(x,y,batch_size)
        
        len(mini_batches)
        
        return mini_batches,cost_array, param, Z_all, A_all, dZ_all, dW_all ,Vdw,Vdb,Sdw,Sdb
 
    def create_mini_batch(self,x_train,y_train,batch_size):
        ## create mini bacthes creation begins
        mini_batch_size = batch_size
        mini_batches = []
        u = x_train.shape[1]
        
        perm = list(np.random.permutation(u))
        shuffled_x = x_train[:,perm]
        shuffled_y = y_train[:,perm]
        
        num_min_batches = int(np.floor(u/mini_batch_size))
        
        for k in range(0,num_min_batches):
            mini_batch_x = shuffled_x[:,k*mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_y = shuffled_y[:, k*mini_batch_size:(k+1)*mini_batch_size]
            
            mini_batch=(mini_batch_x, mini_batch_y)
            mini_batches.append(mini_batch)

        ## handling the remaning datapoints
        if u % mini_batch_size  != 0:
            mini_batch_x = shuffled_x[:, num_min_batches*mini_batch_size: ]
            mini_batch_y = shuffled_y[:, num_min_batches*mini_batch_size: ]
            mini_batch = (mini_batch_x, mini_batch_y)
            mini_batches.append(mini_batch)
        return mini_batches
        
    def run_nn_epochs(self,Z_all, A_all,param,cost_array,dZ_all,dW_all,Vdw,Vdb,nw_size,layer_nn,learning_rate,mini_batches,beta1,beta2,epoch_num,gradient,data_size,batch_size):
        
        num_batches = len(mini_batches)
        alpha = learning_rate
        cost_epoch_array=[]
        auc_array=[]
        
        for epoch in range(1,epoch_num+1):
            cost = 0
            

            for num in range(0,num_batches):

                x_min = mini_batches[num][0] 
                y_min = mini_batches[num][1]     
                
                Z_all, A_all = self.forward_prop(param,x_min,nw_size)
                A_all['A'+str(0)]=x_min
                
                temp = self.comp_cost(A_all,y_min,nw_size)
                if (np.isnan(temp) == True or np.isinf(temp)==True or  np.isneginf(temp) ==True ):
                    print("")
                else : 
                    cost += temp      
                
                dZ_all,dW_all,db_all,Vdw_corrected,Vdb_corrected,Sdw_corrected,Sdb_corrected = self.backward_prop(layer_nn,A_all,y_min,param,Z_all,beta1,beta2,gradient)
                
                param = self.param_update(dW_all,db_all,Vdw_corrected,Vdb_corrected,Sdw_corrected,Sdb_corrected,param,alpha,nw_size,gradient)
                            
            cost = cost/batch_size  
            cost_array.append(cost) 
            
            ## Implement Grdient Clipping!!!!
            
            if(epoch % 100 == 0):
                cost_epoch_array.append(cost)
           
        return (cost_epoch_array,cost_array,auc_array,Z_all,A_all,dZ_all,dW_all,Vdw,Vdb, param)        
    
    def relu(self,x):
        return (x>0) * x
  
    def sigmoid(self,x):
        
        sig= 1/(1+ np.exp(- x))
        
        return sig  
    
    def relu_deriv(self,x):
        print(x)
        return x>0
    
    def forward_prop(self, param,x_min,nw_size):
        size_nn = nw_size
        A = x_min

        A_prev = A   
        A_all ={}
        Z_all = {}    
        
        for i in range (1,size_nn-1):
            W = param['W'+str(i)]
            A_prev = A
            b = param['b' + str(i)]
   
            Z = np.dot(W,A_prev) + b 
            A = self.relu(Z)
        
            ### Implemented Dropout with 70% probability
            dropout_mask = np.random.rand(A.shape[0],A.shape[1]) < .7 ## 30% neurons will be switched off 
            A *= dropout_mask
        
            Z_all['Z' + str(i)] = Z
            A_all['A' + str(i)] = A    
            
        ## calculate output layer Z & A using Sigmoid ##   
        W = param['W'+str(size_nn - 1)]
        A_prev = A_all['A' + str(size_nn-2)]
        b = param['b' + str(size_nn - 1)]
        
        Z= np.dot(W,A_prev) + b
        A = self.sigmoid(Z)
        
        
        ### clipping predictions between .0001 and .9999
        A = np.where(A == 1.0, .9999,A)
        A = np.where(A == 0.0, .0001,A)
        
        
        Z_all['Z' + str(size_nn-1)] = Z
        A_all['A'+str(size_nn-1)] = A
            
        return (Z_all, A_all)    
        
    def comp_cost(self,A_all,y_min,size_nn):
    
        y_hat = A_all['A'+str(size_nn-1)]
        y_act = y_min
        m = np.size(y_min)
        
        cost = - np.sum((y_act*np.log(y_hat)) + (1-y_act)*np.log(1-y_hat)) / m
        np.squeeze(cost)
            
        return cost 
    
    def backward_prop(self,layer_nn,A_all,y_min,param,Z_all,beta1,beta2,gradient):
        layer_nn = self.layer_nn
        t=2
        size_nn = len(layer_nn)
        m = np.size(y_min)
        dZ_all,dW_all, db_all = {},{},{}
        Vdw, Vdb, Sdw, Sdb = {},{},{},{}
        Vdw_corrected,Vdb_corrected,Sdw_corrected,Sdb_corrected = {},{},{},{}
              
        dz = A_all['A'+str(size_nn-1)] - y_min
        dw = (np.dot(dz,A_all['A'+str(size_nn-2)].T))/m
        db = (np.sum(dz, axis=1, keepdims=True))/m
        dZ_all['dZ' + str(size_nn-1)] = dz
        dW_all['dW' + str(size_nn-1)] = dw
        db_all['db' + str(size_nn-1)] = db
            
        for i in range(size_nn-2,0,-1):        
            dz = np.dot(param['W'+str(i+1)].T,dZ_all['dZ' + str(i+1)])
            dz = dz*self.relu_deriv(Z_all['Z' + str(i)])
            dw = np.dot(dz,A_all['A' + str(i-1)].T)/m
            db = np.sum(dz,axis=1,keepdims=True)/m        
            dZ_all['dZ' + str(i)] = dz
            dW_all['dW' + str(i)] = dw
            db_all['db' + str(i)] = db    
        
        for i in range(1,size_nn):
            ## initializaing Gradient Momemtum Parameters
            Vdw["dW" + str(i)] = np.zeros_like(dW_all["dW" + str(i)])
            Vdb["db" + str(i)] = np.zeros_like(db_all["db" + str(i)])   
            Sdw["dW" + str(i)] = np.zeros_like(dW_all["dW" + str(i)])
            Sdb["db" + str(i)] = np.zeros_like(db_all["db" + str(i)])
        
        
        if (gradient == "GDM"):            
             for i in range(1,size_nn):
                Vdw["dW"+ str(i)] = beta1*Vdw["dW"+ str(i)] + (1-beta1)*dW_all["dW" + str(i)]
                Vdb["db"+ str(i)] = beta1*Vdb["db"+ str(i)] + (1-beta1)*db_all["db" + str(i)]
                Vdw_corrected["dW" + str(i)] = Vdw["dW"+ str(i)] / (1-np.power(beta1,t))
                Vdb_corrected["db" + str(i)] = Vdb["db"+ str(i)] / (1-np.power(beta1,t))
                
        elif (gradient == "RMSprop"):             
             for i in range(1,size_nn):
                Sdw["dW"+ str(i)] = beta2*Sdw["dW"+ str(i)] + (1-beta2)*np.square(dW_all["dW" + str(i)])
                Sdb["db"+ str(i)] = beta2*Sdb["db"+ str(i)] + (1-beta2)*np.square(db_all["db" + str(i)])
                Sdw_corrected["dW" + str(i)] = Sdw["dW"+ str(i)] / (1-np.power(beta2,t))
                Sdb_corrected["db" + str(i)] = Sdb["db"+ str(i)] / (1-np.power(beta2,t))
                
        elif (gradient == "Adam"):
             for i in range(1,size_nn):
                Vdw["dW"+ str(i)] = beta1*Vdw["dW"+ str(i)] + (1-beta1)*dW_all["dW" + str(i)]
                Vdb["db"+ str(i)] = beta1*Vdb["db"+ str(i)] + (1-beta1)*db_all["db" + str(i)]
                Sdw["dW"+ str(i)] = beta2*Sdw["dW"+ str(i)] + (1-beta2)*np.square(dW_all["dW" + str(i)])
                Sdb["db"+ str(i)] = beta2*Sdb["db"+ str(i)] + (1-beta2)*np.square(db_all["db" + str(i)])
                
                Vdw_corrected["dW" + str(i)] = Vdw["dW"+ str(i)] / (1-np.power(beta1,t))
                Vdb_corrected["db" + str(i)] = Vdb["db"+ str(i)] / (1-np.power(beta1,t))
                Sdw_corrected["dW" + str(i)] = Sdw["dW"+ str(i)] / (1-np.power(beta2,t))
                Sdb_corrected["db" + str(i)] = Sdb["db"+ str(i)] / (1-np.power(beta2,t))
                
        else:
             raise Exception ('User selected wrong gradient descent optimizer')
             
        
        return (dZ_all,dW_all,db_all,Vdw_corrected,Vdb_corrected,Sdw_corrected,Sdb_corrected)
          
    def param_update(self,dW_all,db_all,Vdw_corrected,Vdb_corrected,Sdw_corrected,Sdb_corrected,param,alpha,nw_size,gradient):
        
        size_nn = nw_size
        epsilon = .000000001
        if (gradient == "GDM"):            
            for i in range(1,size_nn):
                param['W'+str(i)] -= alpha*Vdw_corrected['dW'+str(i)] # using GDM parameters to updated weight
                param['b'+str(i)] -= alpha*Vdb_corrected['db'+str(i)] # using GDM parameters to updated bias
                
        elif (gradient == "RMSprop"):
            for i in range(1,size_nn):
                param['W'+str(i)] -= (alpha*dW_all['dW'+str(i)]/np.sqrt(Sdw_corrected["dW"+str(i)] + epsilon)) # using GDM parameters to updated weight
                param['b'+str(i)] -= (alpha*db_all['db'+str(i)]/np.sqrt(Sdb_corrected["db"+str(i)] + epsilon)) # using GDM parameters to updated bias
        
        elif (gradient == "Adam"):
            for i in range(1,size_nn):
                param['W'+str(i)] -= (alpha*Vdw_corrected['dW'+str(i)]/np.sqrt(Sdw_corrected["dW"+str(i)] + epsilon)) # using GDM parameters to updated weight
                param['b'+str(i)] -= (alpha*Vdb_corrected['db'+str(i)]/np.sqrt(Sdb_corrected["db"+str(i)] + epsilon)) # using GDM parameters to updated bias
            
        return param
      
    def cost_graph(self,cost_array):
        
        cost_array= np.array(cost_array)
        cost_array = cost_array[np.isfinite(cost_array)]
        xs = np.arange(1,len(cost_array)+1)
        
        ### Cost Graph #### 
        plt.plot(xs,cost_array)
        plt.xlabel('iterations')
        plt.ylabel('Cost Function')
        plt.show()
        print("################################# Cost Graph for training dataset has been plotted ! ################################# ")
        return cost_array

    def train_network(self,x,y, learning_rate=.001, beta1=.9, beta2=.999,batch_size=32,network_size=3,gradient="Adam",epoch_num=1000 ):

        layer_nn = self.layer_nn
        ## feeding the input layer neurons !
        layer_nn.append(int(x.shape[0]))
        
        print("You have choosen a ",network_size+1," layers network, with ",network_size," hidden layers.\n")
        inp = input("If you didn't choose this netowrk size and wish to choose one \nPlease press Y and enter start over \n         #########  or  ######### \nPress N to continue entering the number of neurons for each layer \n")
        
        if (inp == "Y" or inp == "y"):
            raise Exception ('You exit the network as you wanted to choose different network arch !. Please start over ')
          
       
        else:
            for i in range(0,network_size):
                print("Enter number of neurons for hidden layer", i+1 ,":")  
                item = int(input())
                layer_nn.append(item)
                
            ## feeding the output layer neuron !
            layer_nn.append(int(1))
            
            nw_size = len(layer_nn)
            
            mini_batches,cost_array, param, Z_all, A_all, dZ_all, dW_all ,Vdw,Vdb,Sdw,Sdb = self.dnn_preprocessing(x,y,batch_size,layer_nn)
            
            data_size = x.shape[1]
            
            print("Network Modeling started at ", datetime.now())
            cost_epoch_array,cost_array,auc_array,Z_all,A_all,dZ_all,dW_all,Vdw,Vdb, param = self.run_nn_epochs(Z_all, A_all,param,cost_array,dZ_all,dW_all,Vdw,Vdb,nw_size,layer_nn,learning_rate,mini_batches,beta1,beta2,epoch_num,gradient,data_size,batch_size)
            
            ### Cost Graph Function per iteration/epoch
            cost_array = self.cost_graph(cost_array)
            
            ### Cost Graph Function per 100 iterations/epoch
            cost_array = self.cost_graph(cost_epoch_array)
                
            print("\n Training of the network completed at ", datetime.now(), " \n Minimum cost function value in training is ",min(cost_array))
        
        return param   
    
