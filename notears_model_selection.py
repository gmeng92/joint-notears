# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:20:05 2020
This is the model selection part of the model
@author: zgeme
"""

import numpy as np

def _score_func(X, W, thres = 0.2, criteria_type ='AIC'):
    """ score function for different modelselection criteria
    Args:
        -LL, the likelihood function value
        -p, number of variables in the model
        -n, number of samples in the model
        -criteria_type, string of criteria type: AIC, AICc, BIC
    Output:
        -score_val, the score of the model
    """
    def _cal_LL(X, W):
        """Calculate the likelihood for the SEM 
        Args:
            -X, n *p array of the observation
            -W, p*p array of the regression parameter
            Remark: n,p are the sample size and dimension respectively
         Outputs:
            -logLL elihood of the model if noises are assumed as Multinormal
        """
        N= X.shape[0]
        p = X.shape[1]
        # residual matrix 
        R = X-X@W
        # Maximum likelihood estimation of the residual covariance matrix
        Sigma= np.zeros([p, p])
        for n in range(N):
            Sigma = Sigma + R[n,:].reshape([p,1])@R[n,:].reshape([1,p]) 
    
        Sigma = Sigma/N
    
        logLL = -N/2 * (np.log(np.linalg.det(Sigma)) + p)
    
        return logLL
    # let's set the nonzero elements
    num_parameter = (abs(W)>thres).sum()
    N = X.shape[0]
    if criteria_type == 'AIC':
        score_val =  -2*_cal_LL(X,W) + 2*num_parameter
    elif criteria_type == 'AICc':
        k = num_parameter
        score_val = -2*_cal_LL(X,W) + 2*k + (2*k^2+2*k)/(N-k-1)
    elif criteria_type == 'BIC':
        score_val = -2*_cal_LL(X,W) + np.log(N)*num_parameter
    
    return score_val

if __name__ == '__main__':
    import os
    os.chdir('C:\\Users\\zgeme\\Dropbox\\\GitHub\\notears')
    import sys
    sys.path.append('C:\\Users\\zgeme\\Dropbox\\GitHub\\notears')
    sys.path.append('C:\\Users\\zgeme\\Dropbox\\GitHub\\notears\\src')
    import utils as ut
    import notears as nt
    
    ut.set_random_seed(1)
    n, d, s0, graph_type, sem_type = 20, 10, 10, 'ER', 'gauss'
    B_true = ut.simulate_dag(d, s0, graph_type)
    W_true = ut.simulate_parameter(B_true)
    # np.savetxt('W_true.csv', W_true, delimiter=',')

    X = ut.simulate_linear_sem(W_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')
    lambda1_list = [0.0, 0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3]
    aic_list = list()
    for  lambda1 in lambda1_list:
        W_est = nt.notears_linear_l1(X, lambda1= lambda1, 
                                     w_threshold= 0.0, loss_type='l2')
    #assert ut.is_dag(W_est)
    #np.savetxt('W_est.csv', W_est, delimiter=',')
        W_est[abs(W_est)<0.2] = 0
        acc = ut.count_accuracy(B_true, W_est != 0)
        AIC_score = _score_func(X, W_est, criteria_type ='BIC')
        aic_list.append(AIC_score)
        print('k=', (abs(W_est)> 0.2).sum(),'\n')
        print(lambda1, AIC_score,'\n', acc)
    
    #%% visualize the modelselction result
    import matplotlib.pyplot as plt
    plt.plot(lambda1_list, aic_list, 'k', lambda1_list, aic_list, 'ro')
    plt.title('Model selection using BIC', size = 16)
    plt.xlabel('$\lambda_1$')
    plt.ylabel('BIC Score')
    plt.show

