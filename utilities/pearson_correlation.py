from scipy.stats import pearsonr
import math

def choose_with_pearson(n_var: int, X, y):
    """
    retorna
    """
    correlations = []
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations.append(corr)

    # obtendo o módulo dos coeficientes de correlação
    correlations = map(math.fabs, correlations)


    # criar um dicionário de features e seus coeficientes de correlação
    corr_dict = dict(zip(X.columns, correlations))


    # ordena o dicionário por coeficiente de correlação em ordem decrescente
    sorted_corr_dict = {k: v for k, v in sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)}

    N = n_var
    selected_features = list(sorted_corr_dict.keys())[:N]


    x_selected = X[selected_features]
    
    return x_selected, sorted_corr_dict