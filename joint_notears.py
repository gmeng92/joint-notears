# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:14:21 2019
Joint notears
- modified base on the notears @ https://github.com/xunzheng/notears by Xun Zheng.

@author: zgeme
"""

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def joint_notears(X, W_est, lambda1, lambda2, loss_type, max_iter=100, h_tol=2e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W \sum_k L(W_k; X_k) + lambda1 ‖W_k‖_1 s.t. h(W_k) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray):  a list of [n, d] sample matrices with length K
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(W_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [K, d, d] estimated DAGs
    """
    def _loss(W):
        """Evaluate value and gradient of loss.
        Args:
            W (np.array): [K,d,d] weighted adjancy matrices of K group
        
        Returns：
            loss - the smooth loss term: l2, logistic, poisson
            G_loss - gradient of the smooth loss term
        """
        K = len(X)
        M = list()
        R = list()
        G_loss = np.zeros([K,d,d])
        loss = 0
        for k in range(K):
            Mk = X[k] @ W[k, :, :]
            M.append(Mk)    
            Rk = X[k] - Mk
            R.append(Rk)
        # loss types 
        if loss_type == 'l2':
            for k in range(K):             
                lossk = 0.5 / X[k].shape[0] * (R[k] ** 2).sum()
                G_loss[k,:,:] = - 1.0 / X[k].shape[0] * X[k].T @ R[k]
                loss = lossk + loss             
        elif loss_type == 'logistic':
            for k in range(K):
                lossk = 1.0 / X[k].shape[0] * (np.logaddexp(0, M[k]) - X[k] * M[k]).sum()
                G_loss[k,:,:] = 1.0 / X[k].shape[0] * X[k].T @ (sigmoid(M[k]) - X[k])
                loss = loss + lossk
        elif loss_type == 'poisson':
            for k in range(K):
                Sk = np.exp(M[k])
                lossk = 1.0 / X[k].shape[0] * (Sk - X[k] * M[k]).sum()
                G_loss[k,:,:] = 1.0 / X[k].shape[0] * X[k].T @ (Sk - X[k])
                loss = loss + lossk
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss
    
    def _group(W):
        """ group term of the adjacency matrices 
            P_group = \sum_{ i != j} (\sum_{k} (W[k, i, j])^2)^(1/2) 
        Args:
            W (np.array): [K,d,d] weighted adjancy matrices of K group
        
        Returns：
            group_val - the group regularize term
            G_group - gradient of the group regularize term
                
        """
        
         
        K = W.shape[0]
        G_group = np.zeros([K, d, d])
        
        W_hat = np.zeros([d,d])
        for i in range(d):
            for j in range(d):
                if i!= j:
                    W_hat[i,j] = np.linalg.norm(W[:,i,j])
        
        group_val = W_hat.sum()
        W_hatD = W_hat
        W_hatD[W_hat == 0] = 1 # avoid dividing 0
        for k in range(K):
            G_group[k,:,:] = np.divide(W[k,:,:], W_hatD)
            
        return group_val, G_group
            
    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        #     E = slin.expm(W * W)  # (Zheng et al. 2018)
        #     h = np.trace(E) - d
        K = W.shape[0]
        h = np.zeros([K,1])
        G_h = np.zeros([K,d,d])
        for k in range(K):
            Mk = np.eye(d) + W[k,:,:] * W[k,:,:] / d  # (Yu et al. 2019)
            Ek = np.linalg.matrix_power(Mk, d - 1)
            h[k] = (Ek.T * Mk).sum() - d
            G_h[k,:,:] = Ek.T * W[k,:,:] * 2
            
        return h, G_h

    def _adj(w, K):
        """Convert doubled variables ([2 K*d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:K*d*d] - w[K*d*d:]).reshape([K, d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w, K)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        group_val, G_group = _group(W)
        obj = loss + 0.5 * rho * (h * h).sum() + (alpha * h).sum() + lambda1 * w.sum() + lambda2 * group_val
        hterm_G = np.zeros([K, d, d])
        for k in range(K):
            hterm_G[k,:,:] = (rho * h[k]+ alpha[k]) * G_h[k, :, :]
        G_smooth = G_loss + hterm_G + lambda2*G_group
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj
    
    def _initial(W_est):
        """ Initialize the optimization with the notear result """
        """ Convert the length K list W_est  into [2, K*d*d] matrix w_est"""
        w_temp = np.zeros([K,d,d])
        for k in range(K):
            w_temp[k,:,:] = W_est[k,:,:]
        w_temp = w_temp.reshape([K*d*d])
        pos = (w_temp >0)
        neg = (w_temp <0)
        
        w_est = np.zeros([2, K*d*d])
        w_est[0, pos] = w_temp[pos]
        w_est[1, neg] = w_temp[neg]
        return w_est.reshape([2*K*d*d])
        
    K, d = len(X), X[0].shape[1]
    w_est = _initial(W_est)
    rho, alpha, h =  1.0, np.zeros([K,1]), np.inf*np.ones([K,1])  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for k in range(K) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new, K))
            if h_new.sum() > 0.25 * h.sum():
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h.sum() <= h_tol*K or rho >= rho_max:
            break
    W_est = _adj(w_est, K)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

#%%
if __name__ == '__main__':
    import utils_joint as utj
    #utj.set_random_seed(1)

    n, d, s0, K, rho, graph_type, sem_type = 100, 20, 20, 2, 0.2, 'SF', 'gauss'
    B_true_list = utj.simulate_dags(d, s0, K, rho, graph_type)
    W_true_list = []
    for k in range(K):
        W_true = utj.simulate_parameter(B_true_list[k])
        W_true_list.append(W_true)
        
    # np.savetxt('W_true.csv', W_true_list, delimiter=',')
    
    X_list = []
    for k in range(K):
        X = utj.simulate_linear_sem(W_true, n, sem_type)
        X_list.append(X)
        
    # np.savetxt('X.csv', X_list, delimiter=',')
   #%%
    W0 = np.zeros([K,d,d]) 
    W_est = joint_notears(X_list, W0, lambda1=0, lambda2 = 0, loss_type='l2')
    for k in range(K):
        assert utj.is_dag(W_est[k,:,:])
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    # counnt the accuracy for each group member
    acc_list = []
    for k in range(K):
        acc = utj.count_accuracy(B_true_list[k], W_est[k,:,:] != 0)
        acc_list.append(acc)
        print(acc)
        


