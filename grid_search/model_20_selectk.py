import pandas as pd
import numpy as np
import math
from utilities.neuron_architectures import create_architectures, test_alphas
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('data/PCOS_selected_features.csv')

X = data.drop(columns=['PCOS (Y/N)'], axis=1)
y = data["PCOS (Y/N)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train_norm = (X_train - np.min(X_train))/(np.max(X_train) - np.min(X_train))
X_test_norm = (X_test - np.min(X_train))/(np.max(X_train) - np.min(X_train))

n_o = 1
n_i = X.shape[1] + 1

# print(f'No: {n_o}')
# print(f'Ni: {n_i}')

n_h = test_alphas(n_i, n_o, [0.5, 2, 3])

layer_arch = create_architectures([2, 9, 14], n_h)

# print(layer_arch)
# print(len(layer_arch))

param_grid_mlp = {
    'hidden_layer_sizes': layer_arch, 
    'activation': ['relu'],
    'batch_size': [16],
    'solver': ['adam'],
    'beta_1': [0.9, 0.8, 0.95],
    'beta_2': [0.999, 0.95],
    'n_iter_no_change': [30],
    'verbose': [False],
    'max_iter': [500, 700]
}

grid_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, verbose=1, cv=5, n_jobs=-1, scoring='accuracy')

grid_mlp.fit(X_train_norm.values, y_train)


results = pd.DataFrame(grid_mlp.cv_results_)
results.to_csv('data/PCOS_20_selectk.csv')


