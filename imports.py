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
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')



#functions

def clf_scores(clf,X_train, X_test,y_train, y_test, negative:str, positive:str):
    
    #Confusion Matrix:
    
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(14,5))
    
    title_font='16'
    label_size={'size':'13'}
    
    ax1.set_title("Train Data", fontsize=title_font)
    ax2.set_title("Test Data", fontsize=title_font)
    ax1.grid(False)
    ax2.grid(False)
    
    label_font = {'size':'15'}
    ax1.set_xlabel('Predicted Outcome', fontdict=label_size);
    ax1.set_ylabel('Actual Outcome', fontdict=label_size);
    ax2.set_xlabel('Predicted Outcome',fontdict=label_size);
    ax2.set_ylabel('Actual Outcome', fontdict=label_size);
    
    ConfusionMatrixDisplay.from_estimator(clf, X_train, y_train, display_labels=(negative, positive),
                                          cmap='Blues', ax=ax1)
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=(negative, positive),
                                          cmap='Blues', ax=ax2)
    fig.tight_layout()
    plt.show()
    
    ##########
    
    # Scores:
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    train_prec = precision_score(y_train, train_pred, zero_division=0)
    train_recall = recall_score(y_train, train_pred)
    train_f1 =f1_score(y_train, train_pred, zero_division=0)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred)
    test_f1 =f1_score(y_test, test_pred, zero_division=0)
    
    print('Train Data:                                 Test Data:')
    print('Accuracy:  {0:<20}             Accuracy:  {1:<10}'.format(train_acc, test_acc))
    print('Recall:    {0:<20}             Recall:    {1:<10}'.format(train_recall, test_recall))
    print('Precision: {0:<20}             Precision: {1:<10}'.format(train_prec, test_prec))
    print('F1:        {0:<20}             F1:        {1:<10}'.format(train_f1, test_f1))
    
    
    #Classification Report:
    
    print('''
    
    
 Test Data Classification Report:
    ''')
    print(classification_report(y_test, test_pred, zero_division=0, target_names=[negative, positive]))
    
    
    
#save scores to score dataframe 
def save_scores(clf, X_test, y_test, score_df, name:str):
    
    #get scores
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred)
    f1 =f1_score(y_test, pred, zero_division=0)
    
    #create new df entry
    entry = {'Name': name, 'Accuracy': acc, 'Recall': recall, 'Precision': prec, 'F1':f1}
    print('adding: ',entry)
    
    #add entry to df
    return score_df.append(entry, ignore_index = True)

    

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




