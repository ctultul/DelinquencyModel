from __future__ import division
import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import datetime
from time import gmtime
import string
import seaborn as sns
from sklearn import tree
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from roc_curve import my_roc_curve
from sklearn.naive_bayes import MultinomialNB
from pylab import plot, show, savefig, xlim, figure, hold, ylim, legend, boxplot, setp, axes
import statsmodels.api as sm

def edit_cols(df):


# Renaming columns with more meaningful names based on the data dictionary definitions
    old_names = ['label', 'msisdn', 'aon', 'daily_decr90',
                'rental90', 'last_rech_date_ma', 'last_rech_amt_ma',
                 'cnt_ma_rech90', 'fr_ma_rech90', 'sumamnt_ma_rech90',
                 'cnt_loans90','amnt_loans90',  'maxamnt_loans90', 'payback90',
                 'daily_decr30', 'rental30', 'last_rech_date_da', 'cnt_ma_rech30','sumamnt_ma_rech30', 'fr_ma_rech30',
                 'cnt_loans30', 'cnt_da_rech30', 'amnt_loans30', 'maxamnt_loans30', 'payback30', 'cnt_da_rech90']

    new_names =  ['label',  'mobNum',  'ageOnCellNtwrk', 'dailyAmtSpnt90',
                  'avgAcntBal90', 'numDaysTillRchrgM', 'amtLastRchrgM',
                  'numRchrgM90', 'frqMRchrg90',  'totAmtRchrgM90',
                  'numLoans90', 'amtLoans90',  'maxAmtLoans90', 'avgPaybkTmDyIn90',
                  'dailyAmtSpnt30', 'avgAcntBal30', 'numDaysTillRchrgD', 'numTimesMRchrg30', 'totAmtRchrgM30','frqMRchrg30',
                  'numLoans30','numTimesDRchrg30', 'amtLoans30', 'maxAmtLoans30', 'avgPaybkTmDyIn30', 'numTimesDRchrg90']

    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)

    # Result of EDA : dropping columns with strong colinearily
    df.drop(['dailyAmtSpnt30', 'avgAcntBal30', 'numDaysTillRchrgD',
                    'numTimesMRchrg30', 'totAmtRchrgM30', 'frqMRchrg30',
                    'numLoans30','numTimesDRchrg30', 'amtLoans30',
                    'maxAmtLoans30', 'avgPaybkTmDyIn30', 'numTimesDRchrg90'], axis=1, inplace=True)
                    # 'pdate', 'pcircle',
                    # 'medianamnt_ma_rech90', 'medianmarechprebal90', 'fr_da_rech30',
                    # 'fr_da_rech90', 'medianamnt_loans30','medianamnt_loans90',
                    # 'medianamnt_ma_rech30', 'medianmarechprebal30'], axis=1, inplace=True)

    df['dailyAmtSpnt90'] = df['dailyAmtSpnt90'].round(2)

    # df['amtLastRchrgM'].fillna(0, inplace=True)
    # df['numDaysTillRchrgM'].fillna(0, inplace=True)
    # df['frqMRchrg90'].fillna(0, inplace=True)
    # df['amtLastRchrgM'].fillna(0, inplace=True)
    # df['numDaysTillRchrgM'].fillna(0, inplace=True)
    # df['maxAmtLoans90'].fillna(0, inplace=True)
    # df['avgPaybkTmDyIn90'].fillna(0, inplace=True)

    df['amtLastRchrgM'].fillna(0, inplace=True)
    df['numDaysTillRchrgM'].fillna(0, inplace=True)
    df['frqMRchrg90'].fillna(0, inplace=True)
    df['amtLastRchrgM'].fillna(0, inplace=True)
    df['numDaysTillRchrgM'].fillna(0, inplace=True)
    df['maxAmtLoans90'].fillna(0, inplace=True)
    df['avgPaybkTmDyIn90'].fillna(0, inplace=True)


    df['numDaysTillRchrgM'] = df['numDaysTillRchrgM'].astype(int)

    df.drop_duplicates('mobNum', keep='last', inplace=True)
    df.drop(['mobNum'], axis=1, inplace=True)
    #df.drop(['avgAcntBal90'], axis=1, inplace=True)
    print df.info()

    return df

def convert_nans_to_medians(df):
    # Converting NaNs to Medians
    for col in df.columns:
        if isinstance(col[0], int) or isinstance(col[0],float):
            df = df[col].fillna(df[col].median())
    return df

def convert_nans_to_means(df):
    # Converting NaNs to Means
    for col in df.columns:
        if isinstance(col[0], int) or isinstance(col[0],float):
            df = df[col].fillna(df[col].mean())
    return df

def get_data(path):
    df = pd.read_csv(path)
    df = edit_cols(df)
    X = df.drop(['label'], axis=1)
    y = df['label'].values.astype(int)
    return X, y

def dump_models(model):
    with open('model.pkl', 'w') as f:
        pickle.dump(model, f)

def train_test_split_func(X,y):
    # test Train split 80/20 with stratified sampling and random state 4 (this will make the sampling same every time)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
    return X_train, X_test, y_train, y_test

def fake_roc(y_pred, y_test):
    tpr, fpr, thresholds = my_roc_curve(y_pred, y_test)

    fig,ax = plt.subplots()
    ax.plot(fpr, tpr, label='ours', linewidth= 5, alpha = 0.5, marker = 'o')
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC")
    return ax

def RF_model(X_train, y_train):
    '''
    RandomForestClassifier
    Input: pandas dataframes
    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    # Got best num of Trees = 35 from trying 5 - 50 num of trees
    RF = RandomForestClassifier(n_estimators=35, max_features=5, oob_score=True)
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_train)
    y_probas = RF.predict_proba(X_train)[:,1]

    # 10-fold Cross Validation
    accuracy = cross_val_score(RF, X_train, y_train, cv=10).mean()
    precision = cross_val_score(RF, X_train, y_train, cv=10, scoring='precision').mean()
    recall = cross_val_score(RF, X_train, y_train, cv=10, scoring='recall').mean()
    f1 = cross_val_score(RF, X_train, y_train, cv=10, scoring='f1').mean()

    ############
    feature_importances = np.argsort(RF.feature_importances_)
    values_for_graphing = RF.feature_importances_[feature_importances[-1:-10:-1]]

    importances = list(X_train.columns[feature_importances[-1:-10:-1]])
    print "Top features:", importances

    # std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
    # indices = np.argsort(importances)[::-1]

    # Plotting the feature importance
    plt.figure()
    figure = plt.figure(figsize=(10,7))
    plt.title("Feature importance", fontsize=18)
    # fig.suptitle('test title', fontsize=20)
    plt.barh(np.arange(len(values_for_graphing)), values_for_graphing, color="g", align="center")
    plt.yticks(np.arange(len(values_for_graphing)), np.array(importances), fontsize=10) #np.array(columns)[indices]
    plt.xlabel('Score')
    plt.ylim([-1, 12])
    plt.show()


    #print 'RF: {},{},{}, {}'.format(accuracy,precision,recall,f1)
    print "Random Forest scores : {}, {}, {}, {} ".format(accuracy,precision,recall,f1)
    print

    # Plotting the ROC Curve
    fpr_RF, tpr_RF, thresholds = roc_curve(y_train, y_probas)
    plt.plot(fpr_RF, tpr_RF, label='Random Forest')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot for Random Forest")
    plt.show()
    ############

    print 'RF Confusion Matrix: '
    print confusion_matrix(y_train, y_pred )

    return accuracy, precision, recall, f1, y_pred, y_probas #, fpr_RF, tpr_RF


# Modelling with Logistic Regression with a 10 fold Cross Validation
def LR_model(X_train, y_train):

    # Instantiating the Logistic Regression model object
    LR = LogisticRegression()

    # fitting the training data
    LR.fit(X_train, y_train)
    predictions = LR.predict_proba(X_train)
    predictions = predictions[:,1]

    # doing a 10-fold Cross Validation
    accuracy = cross_val_score(LR, X_train, y_train, cv=10 ).mean()
    precision = cross_val_score(LR, X_train, y_train, cv=10, scoring='precision').mean()
    recall = cross_val_score(LR, X_train, y_train, cv=10, scoring='recall').mean()
    f1 = cross_val_score(LR, X_train, y_train, cv=10, scoring='f1').mean()

    # Plotting the ROC Curve
    fpr_LR, tpr_LR, thresholds = roc_curve(y_train, predictions) #y_probas)
    plt.plot(fpr_LR, tpr_LR, label='Logistic Regression')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot for Logistic Regression")
    plt.show()
    ############


    # print 'LR: {},{},{},{}'.format(accuracy,precision,recall,f1)
    print "Logistic Regression scores: {}, {}, {}, {} ".format(accuracy,precision,recall,f1)
    print

    # print 'LR Confusion Matrix: '
    # print confusion_matrix(y_train, predictions )

    return accuracy, precision, recall, f1

# Modelling with KNN with a K value of 5
def KNN_model(X_train, y_train):

    # Instantiating the model object with K value = 5
    KN = KNeighborsClassifier(n_neighbors=5)

    # fitting the training data
    KN.fit(X_train, y_train)
    predictions = KN.predict_proba(X_train)
    predictions = predictions[:,1]

    # doing a 10-fold Cross Validation
    accuracy = cross_val_score(KN, X_train, y_train, cv=10).mean()
    precision = cross_val_score(KN, X_train, y_train, cv=10, scoring='precision').mean()
    recall = cross_val_score(KN, X_train, y_train, cv=10, scoring='recall').mean()
    f1 = cross_val_score(KN, X_train, y_train, cv=10, scoring='f1').mean()

    # Plotting the ROC Curve
    fpr_KN, tpr_KN, thresholds = roc_curve(y_train, predictions) #y_probas)
    plt.plot(fpr_KN, tpr_KN, label='KNN Model')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot for KNN Model")
    plt.show()
    ############


    # print 'KN: {},{},{},{}'.format(accuracy,precision,recall,f1)
    print "K Nearest Neighbor scores: {}, {}, {}, {} ".format(accuracy,precision,recall,f1)
    print

    return accuracy, precision, recall, f1


def DT_model(X_train, y_train, max_depth=7, num_features=3):

    # Instantiating the Decision Tree model object
    DT = DecisionTreeClassifier()

    # fitting the training data
    DT.fit(X_train, y_train)
    predictions = DT.predict_proba(X_train)
    predictions = predictions[:,1]

    # Plotting the ROC Curve
    fpr_DT, tpr_DT, thresholds = roc_curve(y_train, predictions) #y_probas)
    plt.plot(fpr_DT, tpr_DT, label='DecisionTree Model')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot for DecisionTree Mode")
    plt.show()
    ############


    # doing a 10-fold Cross Validation
    accuracy = cross_val_score(DT, X_train, y_train, cv=10).mean()
    precision = cross_val_score(DT, X_train, y_train, cv=10, scoring='precision').mean()
    recall = cross_val_score(DT, X_train, y_train, cv=10, scoring='recall').mean()
    f1 = cross_val_score(DT, X_train, y_train, cv=10, scoring='f1').mean()

    #print 'DT: {},{},{},{}'.format(accuracy,precision,recall,f1)
    print "Decision Tree scores : {}, {}, {}, {} ".format(accuracy,precision,recall,f1)
    print

    return accuracy, precision, recall, f1

def standard_confusion_matrix(y_true, y_predict):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])


def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model.score(X_test, y_test), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict), \
           f1_score(y_test, y_predict)

def GB_model(X_train, y_train):
    '''
    GradientBoostingClassifier
    Input: pandas dataframes
    Output: accuracy, precision, recall, y_pred, y_probas
    '''
    GB = GradientBoostingClassifier()
    GB.fit(X_train,y_train)
    # doing a 10-fold Cross Validation
    accuracy = cross_val_score(GB, X_train, y_train, cv=10, scoring='precision').mean()
    precision = cross_val_score(GB, X_train, y_train, cv=10, scoring='precision').mean()
    recall = cross_val_score(GB, X_train, y_train, cv=10, scoring='recall').mean()
    f1 = cross_val_score(GB, X_train, y_train, cv=10, scoring='f1').mean()

    # print 'KN: {},{},{},{}'.format(accuracy,precision,recall,f1)
    print "Gradient Boosting : {}, {}, {}, {} scores".format(accuracy,precision,recall,f1)
    print

    return accuracy, precision, recall, f1

def predict_off_GB(fitted_model, X_test, y_test):
    y_pred = fitted_model.predict(X_test)
    accuracy = fitted_model.score(X_test, y_test)

    print 'GB: {}'.format(accuracy)


def profit_curve(cost_benefit_matrix, probabilities, y_true):
    thresholds = sorted(probabilities)
    thresholds.append(1.0)
    profits = []
    for threshold in thresholds:
        y_predict = probabilities >= threshold
        confusion_mat = standard_confusion_matrix(y_true, y_predict)
        profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
        profits.append(profit)
    return thresholds, profits

def run_profit_curve(model, costbenefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds, profits = profit_curve(costbenefit, probabilities, y_test)
    return thresholds, profits

def plot_profit_models(models, costbenefit, X_train, X_test, y_train, y_test):
    percentages = np.linspace(0, 100, len(y_test) + 1)
    for model in models:
        thresholds, profits = run_profit_curve(model,
                                               costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        plt.plot(percentages, profits, label=model.__class__.__name__)
    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='lower left')
    # plt.savefig('profit_curve.png')
    plt.show()

def find_best_threshold(models, costbenefit, X_train, X_test, y_train, y_test):
    max_model = None
    max_threshold = None
    max_profit = None
    for model in models:
        thresholds, profits = run_profit_curve(model, costbenefit,
                                               X_train, X_test,
                                               y_train, y_test)
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model.__class__.__name__
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit


if __name__ == '__main__':

    # X, y = get_data('/Users/Tultul/Downloads/DelinqData_250K.csv')
    X, y = get_data('/Users/Tultul/Downloads/DelinqData50K.csv')
    # X, y = get_data('/Users/Tultul/Downloads/test_delinq_5k.csv')
    # X, y = get_data('/Users/Tultul/Downloads/test_delinq.csv')
    X_train, X_test, y_train, y_test = train_test_split_func(X, y)

    accuracy, precision, recall, f1, y_pred, y_probas = RF_model(X_train, y_train)
    accuracy, precision, recall, f1 = LR_model(X_train, y_train)
    accuracy, precision, recall, f1 = DT_model(X_train, y_train)
    accuracy, precision, recall, f1 = KNN_model(X_train, y_train)
    accuracy, precision, recall, f1 = GB_model(X_train, y_train)
    #
    #print " SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)

    ###################
    # fpr_RF, tpr_RF, thresholds = roc_curve(y_pred, y_probas)
    # print y_pred
    # print y_probas
    # print thresholds
    # plt.plot(fpr_RF, tpr_RF, label='Random Forest')
    # plt.xlabel("False Positive Rate (1 - Specificity)")
    # plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    # plt.title("ROC plot for Random Forest")
    # plt.show()
    ###################

    #dump_models(best_model)

    # costbenefit = np.array([[455, -12.15], [0, 0]])
    #costbenefit = np.array([[0, -1], [-50, 0]])  # TP=0, FP=-1, FN=-50, TN=0
    # models = [RF(), LR(), GBC(), SVC(probability=True)]
    #models = [RF(), LR(), GBC()]
    #models = [DT()]

    # plot_profit_models(models, costbenefit,
    #                    X_train, X_test, y_train, y_test)
    #plot_profit_models(models, costbenefit, X_train, X_test, y_train, y_test)

    # print confusion_matrix(y_train, y_pred)
