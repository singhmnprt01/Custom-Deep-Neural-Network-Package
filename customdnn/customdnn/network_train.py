# Authors: Manpreet Singh <singhmnprt01@gmail.com>
# customdnn Copyright (C) 2020 singhmnprt01@gmail.com
# License: GNU GENERAL PUBLIC LICENSE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys
import warnings


class SplitData:
    """
     This class helps the user to split the data into train and test.
     User needs to input x, y and the Test percentage

    """

    def split_train_test(self, x, y, test_percentage):
        """
         Parameters
        ----------
        x : dataframe
            Input feature set

        y : dataframe(0,1)
            Target variable

        test_percentage: int 
            Percentage of total data to be used as test
        """

        test_ratio = test_percentage/100

        x = x.to_numpy()
        y = y.to_numpy()

        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_ratio, stratify=y)

        # Transposing it again to make it fit for the NN design
        x_train = x_train.T
        x_test = x_test.T
        y_train = y_train.T
        y_test = y_test.T
        return x_train, x_test, y_train, y_test


class TrainingDeepNetwork:
    """
    The purpose if the class is to train the user defined neural network 
    and return the updated set parameters(weight- w and bias- b)

    User needs to input x, y and certain set of hyperparameters to create a custom Deep Neural Network

    """

    def __init__(self):
        self.t = 2
        self.epsilon = .000000001

    def __create_mini_batch(self, x_train, y_train, batch_size):
        """ mini bacthes creation begins """

        mini_batch_size = batch_size
        mini_batches = []
        u = x_train.shape[1]

        perm = list(np.random.permutation(u))
        shuffled_x = x_train[:, perm]
        shuffled_y = y_train[:, perm]

        num_min_batches = int(np.floor(u/mini_batch_size))

        for k in range(num_min_batches):
            mini_batch_x = shuffled_x[:, k *
                                      mini_batch_size:(k+1)*mini_batch_size]
            mini_batch_y = shuffled_y[:, k *
                                      mini_batch_size:(k+1)*mini_batch_size]

            mini_batch = (mini_batch_x, mini_batch_y)
            mini_batches.append(mini_batch)

        """ handling the remaning datapoints """
        if u % mini_batch_size != 0:
            mini_batch_x = shuffled_x[:, num_min_batches*mini_batch_size:]
            mini_batch_y = shuffled_y[:, num_min_batches*mini_batch_size:]
            mini_batch = (mini_batch_x, mini_batch_y)
            mini_batches.append(mini_batch)

        return mini_batches

    def __dnn_preprocessing(self, x, y, batch_size, layer_nn):
        """ It prepares the neural network architecture as per the user requirements and pre-processes the data to make it network ready. """

        mini_batches = []
        cost_array = []
        param = {}
        size_nn = len(layer_nn)

        """ xavier initialization of weights """
        for i in range(1, size_nn):

            param['W' + str(i)] = np.random.random((layer_nn[i],
                                                    layer_nn[i-1])) * np.sqrt(2/layer_nn[i-1])
            param['b' + str(i)] = np.zeros((layer_nn[i], 1))

            ### Checker to check the dimensions of weights w & bias b ###
            assert(param['W'+str(i)].shape == (layer_nn[i], layer_nn[i-1]))
            assert(param['b'+str(i)].shape == (layer_nn[i], 1))

        """ Activation and gradient param initialization """
        Z_all, A_all = {}, {}
        dW_all, dZ_all, Vdw, Vdb, Sdw, Sdb = {}, {}, {}, {}, {}, {}

        """ Mini-Batches creation """
        mini_batches = self.__create_mini_batch(x, y, batch_size)

        return mini_batches, cost_array, param, Z_all, A_all, dZ_all, dW_all, Vdw, Vdb, Sdw, Sdb

    def __run_nn_epochs(self, Z_all, A_all, param, cost_array, dZ_all, dW_all, Vdw, Vdb, nw_size, layer_nn, learning_rate, mini_batches, beta1, beta2, epoch_num, gradient, data_size, batch_size, dropout_percentage):
        """
        The function is used to train the network using forward and backward propagation

        """
        count = 100
        prog_bar = 0
        num_batches = len(mini_batches)
        alpha = learning_rate
        cost_epoch_array = []
        auc_array = []
        unitColor = '\033[5;36m\033[5;47m'
        endColor = '\033[0;0m\033[0;0m'

        for epoch in range(1, epoch_num+1):
            cost = 0

            for num in range(num_batches):

                x_min = mini_batches[num][0]
                y_min = mini_batches[num][1]

                Z_all, A_all = self.__forward_prop(
                    param, x_min, nw_size, dropout_percentage)
                A_all['A'+str(0)] = x_min

                temp = self.__comp_cost(A_all, y_min, nw_size)
                if (np.isnan(temp) == True or np.isinf(temp) == True or np.isneginf(temp) == True):
                    pass
                else:
                    cost += temp

                dZ_all, dW_all, db_all, Vdw_corrected, Vdb_corrected, Sdw_corrected, Sdb_corrected = self.__backward_prop(
                    layer_nn, A_all, y_min, param, Z_all, beta1, beta2, gradient)

                param = self.__param_update(dW_all, db_all, Vdw_corrected, Vdb_corrected,
                                            Sdw_corrected, Sdb_corrected, param, alpha, nw_size, gradient)

            cost = cost/batch_size
            cost_array.append(cost)

            if(epoch % 100 == 0):
                cost_epoch_array.append(cost)

            incre = int(50.0 / epoch_num * prog_bar)
            sys.stdout.write('\r')
            if prog_bar != epoch_num - 1:
                sys.stdout.write('|%s%s%s%s| %d%%' % (
                    unitColor, '\033[7m' + ' '*incre + ' \033[27m', endColor, ' '*(49-incre), 2*incre))
            else:
                sys.stdout.write('|%s%s%s| %d%%' % (
                    unitColor, '\033[7m' + ' '*20 + 'COMPLETE!' + ' '*21 + ' \033[27m', endColor, 100))

            prog_bar += 1

        return (cost_epoch_array, cost_array, auc_array, Z_all, A_all, dZ_all, dW_all, Vdw, Vdb, param)

    def relu(self, x):
        return (x > 0) * x

    def sigmoid(self, x):

        sig = np.where(x >= 0, 1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))

        return sig

    def relu_deriv(self, x):
        return x > 0

    def __forward_prop(self, param, x_min, nw_size, drop_perc):
        size_nn = nw_size
        A = x_min

        A_prev = A
        A_all = {}
        Z_all = {}

        for i in range(1, size_nn-1):
            W = param['W'+str(i)]
            A_prev = A
            b = param['b' + str(i)]

            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)

            """ Dropout """
            drop_ratio = float(drop_perc/100)
            dropout_mask = np.random.rand(A.shape[0], A.shape[1]) < drop_ratio
            A *= dropout_mask

            Z_all['Z' + str(i)] = Z
            A_all['A' + str(i)] = A

        ## calculate output layer Z & A using Sigmoid ##
        W = param['W'+str(size_nn - 1)]
        A_prev = A_all['A' + str(size_nn-2)]
        b = param['b' + str(size_nn - 1)]

        Z = np.dot(W, A_prev) + b
        warnings.filterwarnings("ignore")
        A = self.sigmoid(Z)

        # clipping predictions between .0001 and .9999 to avoid exploding of gradients
        A = np.where(A == 1.0, .9999, A)
        A = np.where(A == 0.0, .0001, A)

        Z_all['Z' + str(size_nn-1)] = Z
        A_all['A'+str(size_nn-1)] = A

        return (Z_all, A_all)

    def __comp_cost(self, A_all, y_min, size_nn):

        y_hat = A_all['A'+str(size_nn-1)]
        y_act = y_min
        m = np.size(y_min)

        cost = - np.sum((y_act*np.log(y_hat)) + (1-y_act)*np.log(1-y_hat)) / m
        np.squeeze(cost)

        return cost

    def __backward_prop(self, layer_nn, A_all, y_min, param, Z_all, beta1, beta2, gradient):
        size_nn = len(layer_nn)
        m = np.size(y_min)
        dZ_all, dW_all, db_all = {}, {}, {}
        Vdw, Vdb, Sdw, Sdb = {}, {}, {}, {}
        Vdw_corrected, Vdb_corrected, Sdw_corrected, Sdb_corrected = {}, {}, {}, {}

        dz = A_all['A'+str(size_nn-1)] - y_min
        dw = (np.dot(dz, A_all['A'+str(size_nn-2)].T))/m
        db = (np.sum(dz, axis=1, keepdims=True))/m
        dZ_all['dZ' + str(size_nn-1)] = dz
        dW_all['dW' + str(size_nn-1)] = dw
        db_all['db' + str(size_nn-1)] = db

        for i in range(size_nn-2, 0, -1):
            dz = np.dot(param['W'+str(i+1)].T, dZ_all['dZ' + str(i+1)])
            dz = dz*self.relu_deriv(Z_all['Z' + str(i)])
            dw = np.dot(dz, A_all['A' + str(i-1)].T)/m
            db = np.sum(dz, axis=1, keepdims=True)/m
            dZ_all['dZ' + str(i)] = dz
            dW_all['dW' + str(i)] = dw
            db_all['db' + str(i)] = db

        for i in range(1, size_nn):
            # initializaing Gradient Momemtum Parameters
            Vdw["dW" + str(i)] = np.zeros_like(dW_all["dW" + str(i)])
            Vdb["db" + str(i)] = np.zeros_like(db_all["db" + str(i)])
            Sdw["dW" + str(i)] = np.zeros_like(dW_all["dW" + str(i)])
            Sdb["db" + str(i)] = np.zeros_like(db_all["db" + str(i)])

        if (gradient == "GDM"):
            for i in range(1, size_nn):
                Vdw["dW" + str(i)] = beta1*Vdw["dW" + str(i)] + \
                    (1-beta1)*dW_all["dW" + str(i)]
                Vdb["db" + str(i)] = beta1*Vdb["db" + str(i)] + \
                    (1-beta1)*db_all["db" + str(i)]
                Vdw_corrected["dW" + str(i)] = Vdw["dW" +
                                                   str(i)] / (1-np.power(beta1, self.t))
                Vdb_corrected["db" + str(i)] = Vdb["db" +
                                                   str(i)] / (1-np.power(beta1, self.t))

        elif (gradient == "RMSprop"):
            for i in range(1, size_nn):
                Sdw["dW" + str(i)] = beta2*Sdw["dW" + str(i)] + \
                    (1-beta2)*np.square(dW_all["dW" + str(i)])
                Sdb["db" + str(i)] = beta2*Sdb["db" + str(i)] + \
                    (1-beta2)*np.square(db_all["db" + str(i)])
                Sdw_corrected["dW" + str(i)] = Sdw["dW" +
                                                   str(i)] / (1-np.power(beta2, self.t))
                Sdb_corrected["db" + str(i)] = Sdb["db" +
                                                   str(i)] / (1-np.power(beta2, self.t))

        elif (gradient == "Adam"):
            for i in range(1, size_nn):
                Vdw["dW" + str(i)] = beta1*Vdw["dW" + str(i)] + \
                    (1-beta1)*dW_all["dW" + str(i)]
                Vdb["db" + str(i)] = beta1*Vdb["db" + str(i)] + \
                    (1-beta1)*db_all["db" + str(i)]
                Sdw["dW" + str(i)] = beta2*Sdw["dW" + str(i)] + \
                    (1-beta2)*np.square(dW_all["dW" + str(i)])
                Sdb["db" + str(i)] = beta2*Sdb["db" + str(i)] + \
                    (1-beta2)*np.square(db_all["db" + str(i)])

                Vdw_corrected["dW" + str(i)] = Vdw["dW" +
                                                   str(i)] / (1-np.power(beta1, self.t))
                Vdb_corrected["db" + str(i)] = Vdb["db" +
                                                   str(i)] / (1-np.power(beta1, self.t))
                Sdw_corrected["dW" + str(i)] = Sdw["dW" +
                                                   str(i)] / (1-np.power(beta2, self.t))
                Sdb_corrected["db" + str(i)] = Sdb["db" +
                                                   str(i)] / (1-np.power(beta2, self.t))

        else:
            raise Exception(
                'User selected the wrong gradient descent optimizer')

        return (dZ_all, dW_all, db_all, Vdw_corrected, Vdb_corrected, Sdw_corrected, Sdb_corrected)

    def __param_update(self, dW_all, db_all, Vdw_corrected, Vdb_corrected, Sdw_corrected, Sdb_corrected, param, alpha, nw_size, gradient):

        size_nn = nw_size
        if (gradient == "GDM"):
            for i in range(1, size_nn):

                param['W'+str(i)] -= alpha*Vdw_corrected['dW'+str(i)]

                param['b'+str(i)] -= alpha*Vdb_corrected['db'+str(i)]

        elif (gradient == "RMSprop"):
            for i in range(1, size_nn):

                param['W'+str(i)] -= (alpha*dW_all['dW'+str(i)] /
                                      np.sqrt(Sdw_corrected["dW"+str(i)] + self.epsilon))

                param['b'+str(i)] -= (alpha*db_all['db'+str(i)] /
                                      np.sqrt(Sdb_corrected["db"+str(i)] + self.epsilon))

        elif (gradient == "Adam"):
            for i in range(1, size_nn):
                param['W'+str(i)] -= (alpha*Vdw_corrected['dW'+str(i)]/np.sqrt(
                    Sdw_corrected["dW"+str(i)] + self.epsilon))

                param['b'+str(i)] -= (alpha*Vdb_corrected['db'+str(i)]/np.sqrt(
                    Sdb_corrected["db"+str(i)] + self.epsilon))

        return param

    def __cost_graph(self, cost_array, cost_epoch_array):

        cost_array = np.array(cost_array)
        cost_array = cost_array[np.isfinite(cost_array)]
        xs = np.arange(1, len(cost_array)+1)

        ### Cost Graph ####
        plt.plot(xs, cost_array)
        plt.xlabel('No. of Iterations')
        plt.ylabel('Cost Function')
        plt.show()
        print("\n\n################ Cost Graph for training dataset has been plotted ! ################ \n")

        cost_epoch_array = np.array(cost_epoch_array)
        cost_epoch_array = cost_epoch_array[np.isfinite(cost_epoch_array)]
        xs = np.arange(1, len(cost_epoch_array)+1)

        ### Cost Graph ####
        plt.plot(xs, cost_epoch_array)
        plt.xlabel('Per 100 Iterations')
        plt.ylabel('Cost Function')
        plt.show()
        print("################ Cost Graph per 100 iterations for training dataset has been plotted ! ################")

        return cost_array, cost_epoch_array

    def train_network(self, x, y, learning_rate=.001, beta1=.9, beta2=.999, batch_size=32, network_size=3, gradient="Adam", epoch_num=1000, dropout_percentage=70):
        """
        This is the main function of the class which controls other paramount functions, process user input, 
        display cost function graphs and returns trained  set of weight and bias parameters.

        Parameters
        ----------
        x : numpy array
            Input feature set data

        y : numpy array
            Target variable

        learning_rate : float, default = .001
            The rate of learning at which the gradient steps will be taken to minimise the cost

        beta1 : float, default = .9
            Beta constant for Gradient Descent Momentum Optimisation Algorithm

        beta2 : float, default = .999
            Beta constant for Root Mean Square prop Optimisation Algorithm

        batch_size : int, default = 32
            To create customised mini-batches to amplify the processing and improve model accuracy/generalisation/learning.

        network_size : int, default = 3
            A custom variable to design the number of layer of your network. It is exclusive of input and output layer

        gradient : string, default = "Adam"      
            Gradient Descent Optimisation algorithm choosing field. You can input any of the following three optimizers :-
                  GDM : Gradient Descent Momentum
                  RMSprop : Room Mean Square Prop
                  Adam : Adaptive Momentum Estimation

        epoch_num : int, default = 1000
            Number of epochs/iterations for the network. 

        dropout_percentage : int, default = 70
            Percent of neurons to be fused.


        Returns
        -------
        param : dict
            parameter dictionary of trained network.

        """

        layer_nn = []
        # feeding the input layer neurons !
        layer_nn.append(int(x.shape[0]))

        if (int(network_size) == 3):
            print("You have choosen the default network having  ",
                  network_size+1, " layers with ", network_size, " hidden layers.\n")

        else:
            print("You have designed the network having  ",
                  network_size+1, " layers with ", network_size, " hidden layers.\n")

        inp = input("If you wish to update the network design, then Enter N to start over \n         #########  or  ######### \nEnter any other key to continue entering the number of neurons for each layer \n")

        if (inp == "N" or inp == "n"):
            raise Exception(
                'You exit the network as you wanted to choose different network arch !. Please start over ')

        else:
            for i in range(0, network_size):
                print("Enter number of neurons for hidden layer", i+1, ":")
                item = int(input())
                layer_nn.append(item)

            # feeding the output layer neuron !
            layer_nn.append(int(1))

            nw_size = len(layer_nn)

            mini_batches, cost_array, param, Z_all, A_all, dZ_all, dW_all, Vdw, Vdb, Sdw, Sdb = self.__dnn_preprocessing(
                x, y, batch_size, layer_nn)

            data_size = x.shape[1]

            print("Network Modeling started at ", datetime.now(), "\n")
            cost_epoch_array, cost_array, auc_array, Z_all, A_all, dZ_all, dW_all, Vdw, Vdb, param = self.__run_nn_epochs(
                Z_all, A_all, param, cost_array, dZ_all, dW_all, Vdw, Vdb, nw_size, layer_nn, learning_rate, mini_batches, beta1, beta2, epoch_num, gradient, data_size, batch_size, dropout_percentage)

            # Cost Graph Function per iteration/epoch
            cost_array, cost_epoch_array = self.__cost_graph(
                cost_array, cost_epoch_array)

            print("\n Training of the network completed at ", datetime.now(
            ), " \n Minimum cost function value in training is ", min(cost_array))

        return param
