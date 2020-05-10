# Authors: Manpreet Singh <singhmnprt01@gmail.com>
# customdnn Copyright (C) 2020 singhmnprt01@gmail.com
# License: GNU GENERAL PUBLIC LICENSE


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from customdnn.network_train import TrainingDeepNetwork


class PredictDeepNetwork:

    NETWORK_CONSTANT = 2

    def __inti__(self):
        self.param = {}

    def predict_proba(self, x_test, param, network_size=3):
        """

        Parameters
        ----------
        x_test : numpy array
            Testing feature set data.

        param : dict
            Parameter dictionary of trained network.

        network_size : int, default = 3
            Same as trained network size.


        Returns
        -------
        y_test_hat_df : dataframe
            Predicted and Predicted Probabilities of the x_test input data.

        """
        self.param = param
        d = TrainingDeepNetwork()
        nw_size = network_size + self.NETWORK_CONSTANT
        Z_test_All, A_test_all = {}, {}
        Z_test_all, A_test_all = self.__forward_prop_test(
            param, x_test, nw_size, d)
        y_test_hat = A_test_all['A'+str(nw_size-1)]
        y_test_hat_df = pd.DataFrame(y_test_hat.ravel(), columns=['Pred_Prob'])
        y_test_hat_df.loc[y_test_hat_df.Pred_Prob > .5, 'Predicted'] = 1
        y_test_hat_df.loc[y_test_hat_df.Pred_Prob <= .5, 'Predicted'] = 0

        return y_test_hat_df

    def __forward_prop_test(self, param, x, size_nn, d):
        A = x
        A_prev = A
        A_all = {}
        Z_all = {}

        for i in range(1, size_nn-1):
            W = param['W'+str(i)]
            A_prev = A
            b = param['b' + str(i)]

            Z = np.dot(W, A_prev) + b
            A = d.relu(Z)

            Z_all['Z' + str(i)] = Z
            A_all['A' + str(i)] = A

        ## calculate output layer Z & A using Sigmoid ##
        W = param['W'+str(size_nn - 1)]
        A_prev = A_all['A' + str(size_nn-2)]
        b = param['b' + str(size_nn - 1)]

        Z = np.dot(W, A_prev) + b

        A = d.sigmoid(Z)

        # clipping predictions between .0001 and .9999 to avoid nan/inf/-inf in cost computation
        A = np.where(A == 1.0, .9999, A)
        A = np.where(A == 0.0, .0001, A)

        Z_all['Z' + str(size_nn-1)] = Z
        A_all['A'+str(size_nn-1)] = A

        return (Z_all, A_all)

    def nn_auc(self, y_true, y_pred):
        """

        Parameters
        ----------
        y_true : array
            True values obtained after splitting the data
        y_pred : dataframe series
            Predicted value from predict_proba function.User can pass either the predicted value or the pred_prob value

        Returns
        -------
        auc_nn : float
            Area under the curve.
        """

        y_pred = y_pred.to_numpy()
        auc_nn = round(roc_auc_score(
            np.squeeze(y_true), np.squeeze(y_pred)), 3)
        return auc_nn

    def comp_cost_pred(self, y_true, y_pred):
        """

        Parameters
        ----------
        y_true : array
            True values obtained after splitting the data
        y_pred : dataframe series
            Predicted value from predict_proba function. Pass only pre_prob value.

        Returns
        -------
        cost : float
            loss function value for the test data.

        """
        
        y_pred = y_pred.to_numpy()
        m = np.size(y_true)

        cost = - np.sum((np.squeeze(y_true)*np.log(np.squeeze(y_pred))) +
                        (1-np.squeeze(y_true))*np.log(1-np.squeeze(y_pred))) / m
        
        return np.squeeze(cost)
