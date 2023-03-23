import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
import math

def plot_confusion_matrix(y_test, y_pred, percent=False):
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_norm = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    #print(cm_norm)
    
    classes = unique_labels(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    if percent:
      #sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Reds')
      sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Reds')
    else:
      sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax)
    ax.set_xlabel('Previsões')
    ax.set_ylabel('Valores reais')
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes, rotation=0)

    title = 'Matriz de confusão'
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.show()
    
def get_accuracy(y_test, y_pred):
  accuracy = metrics.accuracy_score(y_test, y_pred)
  print(f'ACURÁCIA: {accuracy:.4f}')
  return accuracy
  
def f_score(y_test, y_pred, avg='micro'):
  fscore = metrics.f1_score(y_test, y_pred, average=avg)
  print(f'F-SCORE: {fscore:.4f}')
  return fscore
  
def get_precision(y_test, y_pred):
  prec = metrics.precision_score(y_test, y_pred, average='micro')
  print(f'PRECISÃO: {prec:.4f}')
  return prec

def get_recall(y_test, y_pred):
    recall_s = metrics.recall_score(y_test, y_pred, average='micro')
    print(f"REVOCAÇÃO: {recall_s:.4f}")
    return recall_s