import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('PCOS.csv')
data.dropna(inplace=True)
data = data.replace({'1.99.': 1.99})

del data["Sl. No"]
del data["Patient File No."]
del data["Unnamed: 0"]

data.drop([305], inplace=True)
data['II    beta-HCG(mIU/mL)'] = pd.to_numeric(data['II    beta-HCG(mIU/mL)'])
data['AMH(ng/mL)'] = pd.to_numeric(data['AMH(ng/mL)'])
X = data.drop(columns=['PCOS (Y/N)'], axis=1)
y = data["PCOS (Y/N)"]


correlations = []
for col in X.columns:
    corr, _ = pearsonr(X[col], y)
    correlations.append(corr)

# criar um dicionário de features e seus coeficientes de correlação
corr_dict = dict(zip(X.columns, correlations))

# ordena o dicionário por coeficiente de correlação em ordem decrescente
sorted_corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)}

N = 20
selected_features = list(sorted_corr_dict.keys())[:N]


X_selected = X[selected_features]


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,test_size=0.30)
X_train_norm = (X_train - np.min(X_train))/(np.max(X_train) - np.min(X_train))
X_test_norm = (X_test - np.min(X_train))/(np.max(X_train) - np.min(X_train))

layer_arch = [(1, 2), (2, 1), (1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7), (7, 6), (8, 5), (9, 4), (1, 19), (2, 18), (3, 17), (4, 16), (5, 15), (6, 14), (7, 13), (8, 12), (9, 11), (10, 10), (11, 9), (12, 8), (13, 7), (14, 6)]

param_grid_mlp = {
    'hidden_layer_sizes': layer_arch, 
    'activation': ['relu','logistic'],
    'batch_size': [16, 32],
    'solver': ['adam'],
    'beta_1': [1, 0.9, 0.8],
    'beta_2': [0.999, 0.95, 0.9],
    'n_iter_no_change': [25, 50],
    'verbose': [False],
    'max_iter': [300, 500]
}

grid_mlp = GridSearchCV(MLPClassifier(), param_grid_mlp, verbose=1, cv=5 n_jobs=-1, scoring='accuracy')

grid_mlp.fit(X_train_norm.values, y_train)


results = pd.DataFrame(grid_mlp.cv_results_)
results.to_csv('PCOS_20_attr.csv')