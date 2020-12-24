'''
David Bloomer (13161521)
Birkbeck University, MSc Data Science 18-20 PT
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from hyperopt import hp, STATUS_OK
from hyperopt import Trials, fmin, tpe, space_eval

from joblib import dump, load
from time import time

from b_feat_creation import *

seed = 0

# Pre-Processing, Feature and Model Selection Pipeline
# to determine optimal model based on project objectives evaluating
# performance through speed and metrics

def load_data(folders, file_names, preproc=[], features=[feat_rgb], inc_xpl=True, auto_feat=False, max_files_per_folder=1):
    '''
    Load data for model training based on functionality of load_ml_dataset.
    Once the dataset is loaded, the background and unclassified label are
    removed, and data split in to features and labels using a test-train-split
    strategy, and a test split of 30%.
    ----------------------
    Parameters
    ----------------------
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    feature: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    inc_xpl: Boolean (Default=True)
    auto_feat: Boolean (Default=False)
        Overrides preproc and feature parameters. Requires inc_xpl=True.
    max_files_per_folder: int (Default=-1)
        Number of files to load from each folder. -1 for all files.
    ----------------------
    Returns
    ----------------------
    x: (N*M*0.7,d) ndarray
    x_test: (N*M*0.3,d) ndarray
    y: (N*M*0.7,1) ndarray
    y_test:(N*M*0.3,1) ndarray
    ''' 
    array = load_ml_dataset(folders, file_names, preproc, features, inc_xpl, auto_feat, max_files_per_folder)
    array = array[(array[:,-1] != 0) & (array[:,-1] != 7)]  # drop background and unclassified
    x, y = array[:,:-1], array[:,-1]
    return train_test_split(x, y, stratify=y, test_size=0.3, random_state=seed)

def load_data_imbalance(array, n):
    '''
    Subsample array to give balanced distribution of classes in quantity n.
    Not used in final solution.
    ----------------------
    Parameters
    ----------------------
    array: ndarray
    n: int
        Number of samples required within each class.
    ----------------------
    Returns
    ----------------------
    out: ndarray
    '''
    y = array[:,-1]
    
    classes = []
    
    class_list = list(np.unique(y))
    for c in class_list:
        e = array[(y == c)]
        classes.append((len(e), c, e))

    min_classes = min(classes)[0]
    assert n <= min_classes, f'minimum samples per class ({min_classes}) < n.'
    
    for i, (_, c, e) in enumerate(classes):
        np.random.seed(seed)
        np.random.shuffle(e)
        if i == 0:
            array_output = e[:n]
        else:
            array_output = np.vstack([array_output, e[:n]])
    
    return array_output

def load_models():
    '''
    Classification models for comparison.
    ----------------------
    Returns
    ----------------------
    out: dict
    ''' 
    models = {}
    models['LSVM'] = Pipeline(steps=[('v',VarianceThreshold()),('s',StandardScaler()),('m',model_SVM())])
    # scaling has no impact on tree-based models
    models['RF'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_random_forest())])
    models['RF(W)'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_weighted_random_forest())])
    models['RF15'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_random_forest15())])
    models['RF15(W)'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_weighted_random_forest15())])
    models['BRF'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_balanced_random_forest())])
    models['LGBM'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_lightgbm())])
    models['LGBM(W)'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_weighted_lightgbm())])
    # models['GB'] = Pipeline(steps=[('v',VarianceThreshold()),('m',model_grad_boost())])  # takes 24h+
    return models

def pd_confusion_matrix(conf_mat, labels):
    '''
    Converts ndarray confusion matrix to Pandas DataFrame with labelled axis
    for easier interpretation.
    ----------------------
    Parameters
    ----------------------
    conf_mat: (N,N) ndarray
    labels: list
    ----------------------
    Returns
    ----------------------
    out: pd.DataFrame
    ''' 
    return pd.DataFrame(conf_mat, index=labels, columns=labels)

def evaluate_model(model, x_test, y_test, verbose=False):
    '''
    Model evaluation for test data.
    ----------------------
    Parameters
    ----------------------
    model: class
    x_test: (N,d) ndarray
    y_test: (N,1) ndarray
    verbose: Boolean (Default=False)
    ----------------------
    Returns
    ----------------------
    accuracy: float
    weighted_acc: float
    f1: float
    confusion_matrix: pd.DataFrame
    ''' 
    yhat_test = model.predict(x_test)
    labels = np.unique(yhat_test)
    
    accuracy = metrics.accuracy_score(y_test, yhat_test)
    weighted_acc = metrics.balanced_accuracy_score(y_test, yhat_test)  # average of recall obtained on each class
    f1 = metrics.f1_score(y_test, yhat_test, average='macro', labels=labels)
    weighted_f1 = metrics.f1_score(y_test, yhat_test, average='weighted', labels=labels)
    jaccard = metrics.jaccard_score(y_test, yhat_test, average='macro', labels=labels)
    weighted_jaccard = metrics.jaccard_score(y_test, yhat_test, average='weighted', labels=labels)
    confusion_matrix = metrics.confusion_matrix(y_test, yhat_test, labels=labels)

    if verbose:
        print('Test metrics:')
        print(f'Accuracy: {round(accuracy,2)}')
        print(f'Accuracy (weighted): {round(weighted_acc,2)}')
        print(f'F1: {round(f1,2)}')
        print(f'F1 (weighted): {round(weighted_f1,2)}')
        print(f'Jaccard: {round(jaccard,2)}')
        print(f'Jaccard (weighted): {round(weighted_jaccard,2)}')
    
    return accuracy, weighted_acc, f1, pd_confusion_matrix(confusion_matrix, labels)

def compare_model(folders, file_names, preproc, features, auto_feat=False, rank_idx='none', top_n=30, plot=True):
    '''
    Model comparison for single pre-processing and feature set. Load data for
    model training based on functionality of load_ml_dataset, uses subsample of
    available dataset through max_files_per_folder=1.
    Features can be indexed through combination of rank_idx and top_n, where the
    first n positions within the list are indexed.
    Optional visualisation of model performance expressed through metrics and time.
    ----------------------
    Parameters
    ----------------------
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    feature: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    auto_feat: Boolean (Default=False)
        Overrides preproc and feature parameters. Requires inc_xpl=True.
    rank_idx: str, list (Default='none')
    top_n: int (Default=30)
        Optional indexing of feature array.
    plot: Boolean (default=True)
    ----------------------
    Returns
    ----------------------
    out: pd.DataFrame
        Tabular model performance expressed through metrics and time.
    ''' 
    # load data
    x, x_test, y, y_test = load_data(folders, file_names, preproc, features, inc_xpl=True, auto_feat=auto_feat, max_files_per_folder=1)
 
    # filter top n features
    if rank_idx != 'none':
        x = x[:,rank_idx[:top_n]]
        x_test = x_test[:,rank_idx[:top_n]]
            
    # load models
    models = load_models()

    # evaluate models
    global model_eval  # saved as global variable to retain progress on error
    model_eval = pd.DataFrame(columns=['name',
                                      'acc','acc_w','f1'
                                      't_train','t_test'])
    
    for i, (name, model) in enumerate(models.items()):
        print(name)
        t1 = time()
        model.fit(x, y)
        t2 = time()
        a, a_w, f, _ = evaluate_model(model, x_test, y_test)
        t3 = time()
        model_eval.loc[i] = [name, a, a_w, f, t2-t1, t3-t2]
    
    if plot:
        plot_model_times(model_eval)
        plot_model_performance(model_eval)
    
    return model_eval

def evaluate_all_scenarios():
    '''
    Model comparison for pre-processing and feature set combinations:
    s1) no pre-processing, all features
    s2) Gaussian filter, all features
    s3) Gaussian filter, CLAHE, all features
    s4) no pre-processing, z-normalisation of colour space, all features
    ----------------------
    Returns
    ----------------------
    out: list of pd.DataFrame
        Tabular model performance expressed through metrics and time (for each 
        scenario)
    ''' 
    global s1, s2, s3, s4, s5
    
    features = [feat_greyscale, feat_rgb, feat_hsv, feat_lab,
            feat_sobel, feat_canny, feat_gabor,
            feat_hessian, feat_frangi, feat_sato, feat_meijering,
            feat_lbp2, feat_lbp4, feat_lbp8, feat_lbp16,
            feat_lbp2_uniform, feat_lbp4_uniform, feat_lbp8_uniform, feat_lbp16_uniform]
    
    s1 = compare_model(folders, file_names, [], features, auto_feat=False, rank_idx='none', top_n=12, plot=True)
    s2 = compare_model(folders, file_names, [preproc_gaussian_filt], features, auto_feat=False, rank_idx='none', top_n=12, plot=True)
    s3 = compare_model(folders, file_names, [preproc_gaussian_filt, eq_adaptivehist], features, auto_feat=False, rank_idx='none', top_n=12, plot=True)

    features = [feat_greyscale, feat_rgb, feat_hsv, feat_lab,
            feat_norm_greyscale, feat_norm_rgb, feat_norm_hsv, feat_norm_lab,
            feat_sobel, feat_canny, feat_gabor,
            feat_hessian, feat_frangi, feat_sato, feat_meijering,
            feat_lbp2, feat_lbp4, feat_lbp8, feat_lbp16,
            feat_lbp2_uniform, feat_lbp4_uniform, feat_lbp8_uniform, feat_lbp16_uniform]

    s4 = compare_model(folders, file_names, [], features, auto_feat=False, rank_idx='none', top_n=12, plot=True)
    
    return [s1, s2, s3, s4]

def drop_labels_metrics(array, ref):
    '''
    Drop background and unclassified labels from array to exclude from metric calculation.
    '''
    return array[(ref != 0) & (ref != 7)]

def evaluate_sample(model, preproc=[], features=[feat_rgb], auto_feat=False, rank_idx='none', top_n=12, verbose=False):
    '''
    Evaluate model performance through comparison of input and predicted image
    classification for images taken from each sample. Useful for out of sample
    analysis of model performance. Provides average performance metrics over
    all samples.
    ----------------------
    Parameters
    ----------------------
    model: class
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    feature: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    auto_feat: Boolean (Default=False)
        Overrides preproc and feature parameters.
    rank_idx: str, list (Default='none')
    top_n: int (Default=30)
        Optional indexing of feature array.
    ----------------------
    Returns
    ----------------------
    accuracy: float
    weighted_acc: float
    f1: float
    ''' 
    # define test images
    test_dim = 400
    test_images = [(folders[0], '3', 1100, 1000),
                   (folders[1], '10', 200, 1400),
                   (folders[2], '12', 550, 400),
                   (folders[3], '4', 500, 500)]

    accuracy, weighted_acc, f1, weighted_f1, jaccard, weighted_jaccard = 0, 0, 0, 0, 0, 0
    
    fig = plt.figure(figsize=(8, 4))
    
    for i, (folder, file_name, x, y) in enumerate(test_images):
        array = load_ml_file(folder, file_name, preproc, features, inc_xpl=True, auto_feat=auto_feat, crop=True, x_offset=x, y_offset=y, crop_img_length=test_dim)
        
        x = array[:,:-1]
        y = array[:,-1]
        if rank_idx != 'none':
            assert not auto_feat, "rank_idx must be set to 'none' when auto_feat=True"
            x = x[:,rank_idx[:top_n]]
        yhat = model.predict(x)
        
        y_img, yhat_img = array_restore(y, (test_dim,test_dim)), array_restore(yhat, (test_dim,test_dim))

        y, yhat = drop_labels_metrics(y, y), drop_labels_metrics(yhat, y)
        y_labels = np.unique(y)
        
        accuracy += metrics.accuracy_score(y, yhat)
        weighted_acc += metrics.balanced_accuracy_score(y, yhat)  # average of recall obtained on each class
        f1 += metrics.f1_score(y, yhat, average='macro', labels=y_labels)
        weighted_f1 += metrics.f1_score(y, yhat, average='weighted', labels=y_labels)
        jaccard += metrics.jaccard_score(y, yhat, average='macro', labels=y_labels)
        weighted_jaccard += metrics.jaccard_score(y, yhat, average='weighted', labels=y_labels)
        
        for j, img in enumerate([y_img, yhat_img]):
            ax = fig.add_subplot(2, 4, (i*2)+(j+1))
            plt.imshow(clf2rgb(img))
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    
    fig.tight_layout()
    plt.show()
    
    if verbose:
        print('Average test image metrics:')
        print(f'Accuracy: {round(accuracy/4,2)}')
        print(f'Accuracy (weighted): {round(weighted_acc/4,2)}')
        print(f'F1: {round(f1/4,2)}')
        print(f'F1 (weighted): {round(weighted_f1/4,2)}')
        print(f'Jaccard: {round(jaccard/4,2)}')
        print(f'Jaccard (weighted): {round(weighted_jaccard/4,2)}')
    
    return accuracy/4, weighted_acc/4, f1/4

# Recursive Feature Elimination

def select_rfe(model, folders, file_names, preproc, features):
    '''
    Recursive feature elimination applied to single model.
    ----------------------
    Parameters
    ----------------------
    model: class
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    features: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    ----------------------
    Returns
    ----------------------
    out: list
        Ordered importance of features by index position in x array.
    ''' 
    # load data
    x, x_test, y, y_test = load_data(folders, file_names, preproc, features, inc_xpl=True, max_files_per_folder=1)
    
    # recursive feature elimination
    selector = RFE(model, n_features_to_select=1, step=1)
    selector.fit(x, y)
    rank = selector.ranking_.copy()
    rank_idx = [x for _,x in sorted(zip(rank,range(len(rank))))]

    return rank_idx

def evaluate_rfe(model, folders, file_names, preproc, features, rank_idx, rank_max=15, plot=True):
    '''
    Evaluate incremential model performance of recursive feature elimination up
    to n features (rank_max).
    Optional visualisation of model performance against n features included.
    ----------------------
    Parameters
    ----------------------
    model: class
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    preproc: list (Default: [])
        List of pre-processing steps to apply to image prior to feature creation.
        Equalisation methods are not applied to xpl input.
    features: list (Default: [feat_rgb])
        List of features to create ppl and/or xpl image(s).
    rank_idx: list
    rank_max: int (Default=15)
    plot: Boolean (Default=True)
    ----------------------
    Returns
    ----------------------
    out: list
        Ordered importance of features by index position in x array.
    ''' 
    # load data
    x, x_test, y, y_test = load_data(folders, file_names, preproc, features, inc_xpl=True, max_files_per_folder=1)
    
    # evaluate rfe
    global rfe_eval
    rfe_eval = pd.DataFrame(columns=['top_n_feat',
                                    'acc','acc_w','f1'])

    for i in range(rank_max):
        model = model_lightgbm()
        x = x[:,rank_idx[:i+1]]
        x_test = x_test[:,rank_idx[:i+1]]
        model.fit(x, y)
        a, a_w, f, _ = evaluate_model(model, x_test, y_test)
        rfe_eval.loc[i] = [i+1, a, a_w, f]

    if plot:
        plot_rfe_performance(rfe_eval)
    
    return rfe_eval

def save_model(model, file_out):
    dump(model, file_out)
    print('Model saved.')

def load_model(file_in):
    model = load(file_in)
    print('Load complete.')
    return model

# Hyperparameter Tuning

def lightgbm_hyperparameter_space():
    '''
    Defintion of feature space for LightGBM Classifier: for use in hyperopt library.
    ----------------------
    Returns
    ----------------------
    out: dict
    '''
    return {
            'class_weight': hp.choice('class_weight', [None, 'balanced']),
            'boosting_type_choice': hp.choice('boosting_type',
                                    [{'boosting_type': 'gbdt',
                                         'subsample': hp.uniform('subsample', 0.5, 1)},
                                    {'boosting_type': 'dart'},
                                    {'boosting_type': 'goss',
                                         'subsample': 1.0}
                                    ]),
            
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(1.0)),  # default=0
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(1.0)),  # default=0
            
            'subsample_for_bin': hp.choice('subsample_for_bin', np.arange(50000, 300000+1, 50000, dtype=int)),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),  # default=1.0
        
            'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-3), np.log(1.0)),  # default=1e-3
            'min_child_samples': hp.quniform('min_child_samples', 5, 50, 5),  # default=20
            
            'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 250, 300, 400, 500]),  # default=100
            'num_leaves': hp.choice('num_leaves', np.arange(10, 50+1, dtype=int)),   # default=31
            'max_depth': hp.choice('max_depth', np.arange(3, 30+1, dtype=int)),  # default=-1 (off)
            
            'n_jobs': -1,
            'silent': False
            }

def bayesian_objective_function(params):
    '''
    Objective function for Bayesian hyperparameter: for use in hyperopt library.
    ----------------------
    Returns
    ----------------------
    out: float (0-1)
    '''
    # parameters of dtype=int must have dtype specified
    for param in ['num_leaves','max_depth','min_child_samples']:
        params[param] = int(params[param])
    
    # unnest hierarchical parameters
    for param in params['boosting_type_choice']:
        params[param] = params['boosting_type_choice'][param]
    del params['boosting_type_choice']
    
    model = LGBMClassifier(**params)
    model.fit(x, y)
    
    yhat_test = model.predict(x_test)
    labels = np.unique(yhat_test)
    f1 = metrics.f1_score(y_test, yhat_test, average='macro', labels=labels)
    weighted_f1 = metrics.f1_score(y_test, yhat_test, average='weighted', labels=labels)
    
    metric = (f1 + weighted_f1) / 2
    loss = 1 - metric  # optimisation through minimisation
    
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def hyperparameter_tuning(folders, file_names, plot=True):
    '''
    Bayesian hyperparameter tuning.
    Optional visualisation of objective function performance against evaluations.
    ----------------------
    Parameters
    ----------------------
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    plot: Boolean (Default=True)
    ----------------------
    Returns
    ----------------------
    trials: list
        Performance of objective function for each trial.
    best_param: dict
        Model hyperparameters for optimal evaluation.
    '''
    x, x_test, y, y_test = load_data(folders, file_names, inc_xpl=True, auto_feat=True, max_files_per_folder=1)
    
    space = lightgbm_hyperparameter_space()
    trial_log = Trials()  # trial log
    bayes_cv = fmin(fn=bayesian_objective_function,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trial_log)

    # evaluation
    trials = [1-x for x in trial_log.losses()]
    best_param = space_eval(space, bayes_cv)
    
    if plot:
        plot_tuning_performance(trials)
    
    return trials, best_param

# Model Evaluation

def cross_validation_eval(folders, file_names):
    '''
    K-Fold (k=5) cross validation of tuned model.
    ----------------------
    Parameters
    ----------------------
    folders: list of directory paths
    file_names: list of tuples containing image numbers
    ----------------------
    Returns
    ----------------------
    cross_val_eval: pd.DataFrame
        Tabular performance for each fold expressed through metrics.
    conf_matrix: list
        Confusion matrix for each fold.
    '''
    x, x_test, y, y_test = load_data(folders, file_names, inc_xpl=True, auto_feat=True, max_files_per_folder=1)
    
    folds = KFold(5, shuffle=True, random_state=seed)
    
    global cross_val_eval  # saved as global variable to retain progress on error
    cross_val_eval = pd.DataFrame(columns=['fold',
                                    'acc','acc_w','f1'])
    conf_matrix = []
    
    for fold, (train_idx, test_idx) in enumerate(folds.split(x)):
        print(f'Fold: {fold}')
        model = model_lightgbm_tune()
        model.fit(x[train_idx,:], y[train_idx])
        a, a_w, f, cm = evaluate_model(model, x[test_idx,:], y[test_idx])
        cross_val_eval.loc[fold] = [fold+1, a, a_w, f]
        conf_matrix.append(cm)
    
    return cross_val_eval, conf_matrix

# Classification Models
# wrapper functions for scikit-learn, imblearn and lightgbm functionality

def model_SVM():
    return SGDClassifier(max_iter=100, verbose=0, n_jobs=-1, random_state=seed)

def model_SVM500():
    return SGDClassifier(max_iter=500, verbose=0, n_jobs=-1, random_state=seed)

def model_random_forest():
    return RandomForestClassifier(n_estimators=100, verbose=0, n_jobs=-1, random_state=seed)

def model_weighted_random_forest():
    return RandomForestClassifier(n_estimators=100, class_weight='balanced', verbose=0, n_jobs=-1, random_state=seed)

def model_random_forest15():
    return RandomForestClassifier(n_estimators=100, max_depth=15, verbose=0, n_jobs=-1, random_state=seed)

def model_weighted_random_forest15():
    return RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', verbose=0, n_jobs=-1, random_state=seed)

def model_random_forest30():
    return RandomForestClassifier(n_estimators=100, max_depth=30, verbose=0, n_jobs=-1, random_state=seed)

def model_weighted_random_forest30():
    return RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced', verbose=0, n_jobs=-1, random_state=seed)

def model_grad_boost():
    return GradientBoostingClassifier(n_estimators=100, verbose=0, random_state=seed)

def model_balanced_random_forest():
    return BalancedRandomForestClassifier(n_estimators=100, verbose=0, n_jobs=-1, random_state=seed)

def model_lightgbm():
    return LGBMClassifier(n_estimators=100, silent=True, n_jobs=-1, random_state=seed)

def model_weighted_lightgbm():
    return LGBMClassifier(n_estimators=100, class_weight='balanced', silent=True, n_jobs=-1, random_state=seed)

def model_lightgbm_tune():
    return LGBMClassifier(n_estimators=300, boosting_type='gbdt', subsample=0.778, class_weight=None,
                          colsample_bytree=0.948, learning_rate=0.158, max_depth=19, min_child_samples=25,
                          min_child_weight=0.012, num_leaves=42, reg_alpha=1.024e-05, reg_lambda=0.126, subsample_for_bin=300000,
                          silent=True, n_jobs=-1, random_state=seed)

# Reporting

def plot_model_times(dataframe='none'):
    '''
    For reporting: Visualisation of model train and test times.
    '''
    # pre-calculated results for use in reporting
    if type(dataframe) != pd.core.frame.DataFrame:
        dataframe = pd.DataFrame({'name':['LSVM','RF','RF(W)','BRF','LGBM','LGBM(W)'],
                                   't_train':[60.1, 1832.4, 1850.8, 304.2, 341.1, 354.2],
                                   't_test':[14.6, 49.7, 52.2, 37.1, 28.4, 30.2]})
    
    dataframe = dataframe[['name','t_train','t_test']]
    dataframe['t_train'] = dataframe['t_train'] / 60
    dataframe['t_test'] = dataframe['t_test'] / 60
    dataframe = dataframe.set_index('name')
    ax = dataframe.plot(kind='bar')
    ax.set_ylabel('Time (mins)')

def plot_model_performance(dataframe='none'):
    '''
    For reporting: Visualisation of model performance metrics.
    '''
    # pre-calculated results for use in reporting
    if type(dataframe) != pd.core.frame.DataFrame:
        dataframe = pd.DataFrame({'name':['LSVM','RF','RF(W)','BRF','LGBM','LGBM(W)'],
                                  'acc':[0.764, 0.857, 0.855, 0.758, 0.806, 0.694],
                                  'acc_w':[0.470, 0.694, 0.691, 0.809, 0.598, 0.743],
                                  'f1':[0.769, 0.756, 0.754, 0.667, 0.641, 0.595]})
    
    dataframe = dataframe[['name','acc','acc_w','f1']]
    dataframe = dataframe.set_index('name')
    ax = dataframe.plot(kind='bar')
    ax.legend(loc='lower right')

def plot_rfe_performance(dataframe='none'):
    '''
    For reporting: Visualisation of RFE perforance against n features.
    '''
    # pre-calculated results for use in reporting
    if type(dataframe) != pd.core.frame.DataFrame:
        dataframe = pd.DataFrame({'top_n_feat':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                   'acc':[0.609, 0.663, 0.703, 0.769, 0.783, 0.787, 0.790, 0.790,
                                          0.792, 0.795, 0.798, 0.800, 0.801, 0.801, 0.802],
                                   'acc_w':[0.332, 0.398, 0.441, 0.510, 0.545, 0.551, 0.553, 0.555,
                                            0.561, 0.566, 0.570, 0.579, 0.582, 0.582, 0.585],
                                   'f1':[0.556, 0.506, 0.451, 0.534, 0.580, 0.586, 0.589, 0.592,
                                         0.599, 0.605, 0.609, 0.620, 0.623, 0.624, 0.628]})
    
    dataframe = dataframe[['top_n_feat','acc','acc_w','f1']]
    dataframe.set_index('top_n_feat').plot(kind='line')

def plot_tuning_performance(trials='none'):
    '''
    For reporting: Visualisation of optimisation of Bayesian objective function.
    '''
    if type(trials) != list:
        trials = [0.604,0.739,0.736,0.619,0.654,0.640,0.685,0.722,0.660,0.647,0.680,0.679,0.700,
                  0.647,0.755,0.709,0.689,0.743,0.695,0.625,0.757,0.751,0.754,0.748,0.727,0.721,
                  0.739,0.727,0.746,0.655,0.771,0.770,0.775,0.765,0.690,0.752,0.733,0.678,0.686,
                  0.696,0.664,0.765,0.723,0.637,0.749]
        
    plt.scatter(range(len(trials)), trials)
    plt.xlabel('Iterations')
    plt.ylabel('Dice correlation coefficient')

#%%############################################################################

def main():
    '''
    Pseudo-code to demonstrate applied workflow.
    '''
    # define data folders per sample
    folders = [r'data\15125,6',
               r'data\15475,15',
               r'data\15555,209',
               r'data\15597,18']
    file_names = [('1','2','3','4'),
                  ('1','2','6','10','11','12','14'),
                  ('1','7','8','12'),
                  ('1','2','3','4')]

    # example of pre-processing and feature creation strategy
    preproc = [preproc_gaussian_filt]
    features = [feat_greyscale, feat_rgb, feat_hsv, feat_lab,
            feat_sobel, feat_canny, feat_gabor,
            feat_hessian, feat_frangi, feat_sato, feat_meijering,
            feat_lbp2, feat_lbp4, feat_lbp8, feat_lbp16,
            feat_lbp2_uniform, feat_lbp4_uniform, feat_lbp8_uniform, feat_lbp16_uniform]

    # single model comparison
    compare_model(folders, file_names, preproc, features)
    
    # scenario model comparison for selection
    s1, s2, s3, s4 = evaluate_all_scenarios()
           
    # recursive feature elimination
    model = model_lightgbm()
    rank_idx = select_rfe(model)
    # rank_idx = [19,5,18,2,4,15,62,7,9,20,76,63,8,12,24,3,60,11,41,6,78,
    #             65,16,28,32,64,21,13,36,10,49,72,77,0,74,58,61,42,68,37]

    evaluate_rfe(model, rank_idx)
    compare_model(model, preproc, features, rank_idx=rank_idx, top_n=30)
    # same as: compare_model(model, preproc, auto_feat=True)

    # hyperparameter optimisation
    trials, best_param = hyperparameter_tuning(folders, file_names)
    
    # final model evaluation and training
    cross_val_eval, conf_matrix = cross_validation_eval(folders, file_names)
    
    x, x_test, y, y_test = load_data(folders, file_names, inc_xpl=True, auto_feat=True, max_files_per_folder=1)
    model = model_lightgbm_tune()
    model.fit(x, y)
    save_model(model, r'model\multi_min_f.clf')
   
