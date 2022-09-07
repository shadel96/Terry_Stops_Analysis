import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

#import classifiers
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE



#functions

def clf_scores(clf,X_train, X_test,y_train, y_test):
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    train_prec = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred)
    train_f1 =f1_score(y_train, train_pred)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred)
    test_f1 =f1_score(y_test, test_pred)
    
    print('Train Data:                                 Test Data:')
    print('Accuracy:  {0:<20}             Accuracy:  {1:<10}'.format(train_acc, test_acc))
    print('Recall:    {0:<20}             Recall:    {1:<10}'.format(train_recall, test_recall))
    print('Precision: {0:<20}             Precision: {1:<10}'.format(train_prec, test_prec))
    print('F1:        {0:<20}             F1:        {1:<10}'.format(train_f1, test_f1))




    
def plot_confusion(clf, X_train, X_test, y_train, y_test):
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(13,5))
    
    ax1.set_title("Train Data")
    ax2.set_title("Test Data")
    
    
    plot_confusion_matrix(clf, X_train, y_train, 
                          cmap='Blues', ax=ax1)
    plot_confusion_matrix(clf, X_test, y_test, 
                          cmap='Blues', ax=ax2)
    plt.grid(False)
    plt.show()
    
    

def plot_ROC(clf, X_test, y_test):
    # Probability scores for test set
    y_score = clf.decision_function(X_test)
    # False positive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # Seaborn's beautiful styling
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    # Print AUC
    print('AUC: {}'.format(auc(fpr, tpr)))

    # Plot the ROC curve
    plt.figure(figsize=(7, 7))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()




