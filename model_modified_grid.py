import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import math


def preparing_dataset(data):
    data.dropna(inplace=True)
    data = data.replace({'1.99.': 1.99})
    data.drop([305], inplace=True)
    data['II    beta-HCG(mIU/mL)'] = pd.to_numeric(data['II    beta-HCG(mIU/mL)'])
    data['AMH(ng/mL)'] = pd.to_numeric(data['AMH(ng/mL)'])

    return data

def normalize_split_data(data):
    X = data.drop(columns=['PCOS (Y/N)'], axis=1)
    y = data["PCOS (Y/N)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30)
    X_train_norm = (X_train - np.min(X_train))/(np.max(X_train) - np.min(X_train))
    X_test_norm = (X_test - np.min(X_train))/(np.max(X_train) - np.min(X_train))

    return X_train_norm, X_test_norm, y_train, y_test

def get_neurons_quantity(n_i, n_o):
    alphas = [0.5, 2, 3]
    n_h = []

    for alpha in alphas:
        n_h.append(round(alpha * math.sqrt(n_i * n_o)))

    return n_h

def get_architectures(Nh, quant):
    S = [i for i in range(int(Nh))]
    arch_list = []
    for i in range(len(S)):
        for j in range(len(S)):
            if ((S[i], S[j]) not in arch_list) and (i + j == len(S) and (len(arch_list) < quant)):
                arch_list.append((S[i], S[j]))
    
    return arch_list

def get_layers_architectures(n_h, qtd_1, qtd_2, qtd_3):
    layer_arch = []
    layer_0 = get_architectures(n_h[0], qtd_1)
    layer_1 = get_architectures(n_h[1], qtd_2)
    layer_2 = get_architectures(n_h[2], qtd_3)
    layer_arch = layer_0 + layer_1 + layer_2

    return layer_arch

def grid_serach(mlp_params, X_train, y_train):
    
    pd.DataFrame(columns=['hidden_layer_size', 'activation', 'batch_size'])  
    for hidden_layers_size in mlp_params['hidden_layer_sizes']:
        for activation in mlp_params['activation']:
            for batch_size in mlp_params['batch_size']:
                for solver in mlp_params['solver']:
                    for beta_1 in mlp_params['beta_1']:
                        for beta_2 in mlp_params['beta_2']:
                            for n_iter_no_change in mlp_params['n_iter_no_change']:
                                for max_iter in mlp_params['max_iter']:
                                    mlp = MLPClassifier(
                                        hidden_layer_sizes=hidden_layers_size,
                                        activation=activation, 
                                        batch_size=batch_size, 
                                        solver=solver,
                                        beta_1=beta_1, 
                                        beta_2=beta_2, 
                                        n_iter_no_change=n_iter_no_change,
                                        verbose=False, 
                                        max_iter=max_iter)
                                    
                                    mlp.fit(X_train, y_train)

                                        




if __name__ == '__main__':
    data = pd.read_csv('PCOS.csv')
    data = preparing_dataset(data)
    X_train, X_test, y_train, y_test = normalize_split_data(data)
    n_h = get_neurons_quantity(X_train.shape[1] + 1, 1)
    layer_arch = get_layers_architectures(n_h, 2, 12, 16)

    mlp_params = {
        'hidden_layer_sizes': layer_arch, 
        'activation': ['relu','logistic'],
        'batch_size': [16, 32],
        'solver': ['adam'],
        'beta_1': [1, 0.9, 0.8],
        'beta_2': [0.999, 0.95, 0.9],
        'n_iter_no_change': [25, 30],
        'verbose': [False],
        'max_iter': [300]
    }

    grid_serach(mlp_params)


    
    