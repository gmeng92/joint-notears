# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:13:56 2019

Generate the simulation DAG data set:
    
    1, scale-free net (SF)
    2, erdos-renyi net (ER)
    
    with noise distribution as
    

@author: zgeme
"""

# %% load required packages
import os
os.chdir('C:\\Users\\gzhang1\\Documents\\GitHub\\joint-notears')
import sys
sys.path.append('C:\\Users\\gzhang1\\Documents\\GitHub\\joint-notears')
sys.path.append('C:\\Users\\gzhang1\\Documents\\GitHub\\joint-notears\\src')
#%%
if __name__ == '__main__':
    import os
    import joint_notears
    import numpy as np
    import pickle
    import utils_joint as utj
#from bokeh.io import output_notebook
#output_notebook()

cwd = os.getcwd()

#%% All combinations
##
#n = [20,1000]       # observation
#d = [10,20,50,100]  # variable dimension
#degree = [1, 2, 4]
#graph_type_list = ['erdos-renyi' , 'barabasi-albert']
#noise_type_list = ['linear-gauss', 'linear-exp', 'linear-gumbel']
##  graph type : 'erdos-renyi' , 'barabasi-albert'
##  sem type: 'linear-gauss', 'linear-exp', 'linear-gumbel'
#
##%% loop over all combinations
##   generate the random graphs separatedly
#
#for var in d:
#    for deg in degree:
#        for graph_type in graph_type_list:
#            G = utils.simulate_random_dag(var, deg, graph_type) # ground truth graph
#            for obs in n:
#                for noise_type in noise_type_list:
#                    X = utils.simulate_sem(G, obs, noise_type)
#                    file_name = cwd+'\\'+graph_type+'\\'+'simu'+str(obs)\
#                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ noise_type[7:10]
#                    # save the data as pickle file
#                    with open(file_name,'wb') as pickle_file:
#                        pickle.dump(X, pickle_file)
#                    # save the graph as well
#                    graph_name = cwd+'\\'+graph_type+'\\'+'sGraph'+str(obs)\
#                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ noise_type[7:10]
#                    nx.write_gpickle(G, graph_name)
#                    
#                                   
                    
#%% generated grouped data sequences in common setting
datapath = 'C:\\Users\\gzhang1\\Documents\\JointNotearsData\\simu1'    
n = 300       # observation
d = [10, 20, 50, 100]  # variable dimension
degree = [1, 2, 4]
graph_type_list = ['ER' , 'SF']
sem_type_list = ['gauss', 'exp', 'gumbel']
K = 2   # the number of group
rho = 0.2

#%% loop over all combinations
#   generate the random graphs separatedly

for var in d:
    for deg in degree:
        for graph_type in graph_type_list:
            ## generate the groud truth causal structure
            s0 = deg * var
            G_list = utj.simulate_dags(var, s0, K, rho, graph_type) # ground truth graph
            W_list = []
            for k in range(K):
                W = utj.simulate_parameter(G_list[k])
                W_list.append(W)
            ## generate the data sequence according to the causal graph    
            for sem_type in sem_type_list:
                X_list = []
                for G in G_list:
                    X = utj.simulate_linear_sem(W_list[k], n, sem_type)
                    X_list.append(X)
                file_name = datapath+'\\simuX'+graph_type + str(n)\
                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                    # save the data as pickle file
                with open(file_name,'wb') as pickle_file:
                    pickle.dump(X_list, pickle_file)
                    # save the graph as well
                graph_name = datapath+'\\simuG' +graph_type+ str(n)\
                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(graph_name,'wb') as pickle_file:
                    pickle.dump(G_list, pickle_file)
                        
# group_num_list = [2,3,5]                   
                    
#%% generated grouped data sequences in high dimensional setting
datapath = 'C:\\Users\\gzhang1\\Documents\\JointNotearsData\\simu2'    
n = 50       # observation
d = [20, 40, 60, 100]
degree = [1, 2, 4]
graph_type_list = ['ER' , 'SF']
sem_type_list = ['gauss', 'exp', 'gumbel']
K = 2   # the number of group
rho = 0.2

#%% loop over all combinations
#   generate the random graphs separatedly
for var in d:
    for deg in degree:
        for graph_type in graph_type_list:
            ## generate the groud truth causal structure
            s0 = deg * var
            G_list = utj.simulate_dags(var, s0, K, rho, graph_type) # ground truth graph
            W_list = []
            for k in range(K):
                W = utj.simulate_parameter(G_list[k])
                W_list.append(W)
            ## generate the data sequence according to the causal graph    
            for sem_type in sem_type_list:
                X_list = []
                for G in G_list:
                    X = utj.simulate_linear_sem(W_list[k], n, sem_type)
                    X_list.append(X)
                file_name = datapath+'\\simuX'+graph_type + str(n)\
                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                    # save the data as pickle file
                with open(file_name,'wb') as pickle_file:
                    pickle.dump(X_list, pickle_file)
                    # save the graph as well
                graph_name = datapath+'\\simuG' +graph_type+ str(n)\
                    +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(graph_name,'wb') as pickle_file:
                    pickle.dump(G_list, pickle_file)
#                    
#file_name = 'simu'+str(n)+'n_'+str(d)+'_v'+str(degree)+'_d'
#
## write the variables as pickle file 
#with open(file_name, 'wb') as pickle_file:
#    pickle.dump(Graph_list, pickle_file)
#
#
####### read the pickle file    
##with open(file_name, 'rb') as pickle_file:
##    Graph = pickle.load(pickle_file)
#graph_name ='simuG'+str(n)+'n_'+str(d)+'_v'+str(degree)+'_d' 
#nx.write_gpickle(G, graph_name)


# %% save the simulation data to .mat file
                    # first save the simu1 data #
import scipy.io as sio 

datapath = 'C:\\Users\\gzhang1\\Documents\\JointNotearsData\\simu1'    
n = 300       # observation
d = [10, 20, 50, 100]  # variable dimension
degree = [1, 2, 4]
graph_type_list = ['ER' , 'SF']
sem_type_list = ['gauss', 'exp', 'gumbel']
K = 2   # the number of group
rho = 0.2      

for var in d:
    for deg in degree:
        for graph_type in graph_type_list:
            for sem_type in sem_type_list:
                    # save the ground truth graph
                graph_name = datapath+'\\simuG' +graph_type+ str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(graph_name, 'rb') as pickle_file:
                    Graph_list = pickle.load(pickle_file) 
                graph_mat = datapath+'\\simuG' +graph_type+ str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type+'.mat'
                sio.savemat(graph_mat, {'W': Graph_list})
                    # save the ground truth time series
                file_name = datapath+'\\simuX'+graph_type + str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(file_name, 'rb') as pickle_file:
                    X_list = pickle.load(pickle_file)
                ts_name = datapath+'\\simuX'+graph_type + str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type+'.mat'
                sio.savemat(ts_name, {'X_list': X_list})

# %% then save the simu2 data #
                    
datapath = 'C:\\Users\\gzhang1\\Documents\\JointNotearsData\\simu2'    
n = 50       # observation
d = [20, 40, 60, 100]
degree = [1, 2, 4]
graph_type_list = ['ER' , 'SF']
sem_type_list = ['gauss', 'exp', 'gumbel']
K = 2   # the number of group
rho = 0.2

for var in d:
    for deg in degree:
        for graph_type in graph_type_list:
            for sem_type in sem_type_list:
                    # save the ground truth graph
                graph_name = datapath+'\\simuG' +graph_type+ str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(graph_name, 'rb') as pickle_file:
                    Graph_list = pickle.load(pickle_file) 
                graph_mat = datapath+'\\simuG' +graph_type+ str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type+'.mat'
                sio.savemat(graph_mat, {'W': Graph_list})
                    # save the ground truth time series
                file_name = datapath+'\\simuX'+graph_type + str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type
                with open(file_name, 'rb') as pickle_file:
                    X_list = pickle.load(pickle_file)
                ts_name = datapath+'\\simuX'+graph_type + str(n)\
                +'n_'+str(var)+'v_'+str(deg)+'d_'+ sem_type+'.mat'
                sio.savemat(ts_name, {'X_list': X_list})            
                    