#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
import gc
from sklearn.metrics import f1_score
from sklearn.externals import joblib

def save_model(model, model_path):
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.dump(model_path)

class LgbConfig(object):
    def __init__(self):
        self.params = {
            'boosting_type': 'gbdt', # dart goss rf
            'objective': 'binary', # 二分类
            'metric':'binary_logloss', # 可以用auc
            'num_leaves': 50,
            'learning_rate': 0.01,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'bagging_seed': 55,
            'seed': 77,
            'max_bin': 255,
           
            'nthread': -1,
            'max_depth': -1,
            'verbose': 0
        }
    
        self.nfold = 5
        self.seed = 666
        self.num_boost_round = 5000
        self.early_stopping_rounds = 100
        self.verbose_eval = 100
    
        self.flag = 1 # metric 越小越好，如果-1代表越大越好
        self.min_merror = float('Inf')
    
        self.thres = 0.5


class SwordLgbClassifier(object):
    """
    best lgb sword
    """
    def __init__(self, config):
        self.config = config
        self.cv_score = 0
        self.best_rounds = 100

    def use_metric_auc(self, config):
        self.config.params['metric'] = 'auc'
        self.config.flag = -1
        self.config.min_merror *= -1
        self.model = None

    def get_best_thres(self, data, label, score_func = f1_score):
        """
        score_func must have two params in order
        1: true_label
        2: pred_label
        """
        pred_prob = self.model.predict(data)
        best_score = 0
        for i_thres in range(0, 100):
            pred_label = [int(i > (i_thres / 100.0)) for i in pred_prob]
            fs = score_func(label, pred_label)
            if best_score < fs:
                best_score = fs
                self.config.thres = i_thres / 100.0
        print ('best score: %0.2f best_thres: %0.2f' % (best_score, self.config.thres))

    def get_lgb_dataset(self, data, label, feature_name = "auto", categorical_feature = "auto", weight = None):
        return lgb.Dataset(data, label = label, feature_name = [], categorical_feature = [], weight = weight)

    def load_binary_dataset(self, filepath):
        return lgb.Dataset(filepath)

    def save_binary_dataset(self, data, filepath):
        """
        eg: filepath = train.bin
        """
        data.save_binary(filepath)

    def get_best_rounds_by_cv(self, lgbdata):
        cv_results = lgb.cv(
                            params = self.config.params,
                            train_set = lgbdata,
                            seed = self.config.seed,
                            nfold = self.config.nfold,
                            num_boost_round = self.config.num_boost_round,
                            early_stopping_rounds = self.config.early_stopping_rounds,
                            verbose_eval = self.config.verbose_eval
                           )
        if self.config.flag == -1:
            self.best_rounds = pd.Series(cv_results[self.config.params['metric'] + '-mean']).idxmax()
            self.cv_score = pd.Series(cv_results[self.params['metric'] + '-mean']).max()
        else:
            self.best_rounds = pd.Series(cv_results[self.config.params['metric'] + '-mean']).idxmin()
            self.cv_score = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()
        print ("cv: best rounds:%d cv score:0.4f" % (self.best_rounds, self.cv_score))

    def get_lgb_model(self, lgbdata):
        self.model = lgb.train(self.config.params, lgbdata, num_boost_round = self.best_rounds)
        return True
        
    def find_best_params(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        self.adj_leaves_depth(lgbdata, seed, nfold, early_stopping_rounds)
        self.adj_bin_leafdata(lgbdata, seed, nfold, early_stopping_rounds)
        self.adj_fraction(lgbdata, seed, nfold, early_stopping_rounds)
        self.adj_lambda(lgbdata, seed, nfold, early_stopping_rounds)
        self.adj_eta(lgbdata, seed, nfold, early_stopping_rounds)
        return True
    
    def adj_leaves_depth(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for num_leaves in range(20,200,10):
            for max_depth in range(3,8,1):
                self.config.params['num_leaves'] = num_leaves
                self.config.params['max_depth'] = max_depth 
                cv_results = lgb.cv(
                                    self.config.params,
                                    lgbdata,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.config.flag == -1:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()

                if mean_merror * self.config.flag < self.config.min_merror * self.config.flag:
                    self.config.min_merror = mean_merror
                    best_params['num_leaves'] = num_leaves
                    best_params['max_depth'] = max_depth
        self.config.params.update(best_params)
        
    def adj_bin_leafdata(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for max_bin in range(100,255,10):
            for min_data_in_leaf in range(10,200,10):
                self.config.params['max_bin'] = max_bin
                self.config.params['min_data_in_leaf'] = min_data_in_leaf
                cv_results = lgb.cv(
                                    self.config.params,
                                    lgbdata,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.config.flag == -1:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()

                if mean_merror * self.config.flag < self.config.min_merror * self.config.flag:
                    self.config.min_merror = mean_merror
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
        self.config.params.update(best_params)
        
    def adj_fraction(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for feature_fraction in [0.6,0.7,0.8,0.9]:
            for bagging_fraction in [0.6,0.7,0.8,0.9]:
                self.config.params['feature_fraction'] = feature_fraction
                self.config.params['bagging_fraction'] = bagging_fraction
                cv_results = lgb.cv(
                                    self.config.params,
                                    lgbdata,
                                    seed = seed,
                                    nfold = nfold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    verbose_eval = 0
                                    )
                if self.config.flag == -1:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).max()
                else:
                    mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()

                if mean_merror * self.config.flag < self.config.min_merror * self.config.flag:
                    self.config.min_merror = mean_merror
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
        self.config.params.update(best_params)
    
    def adj_lambda(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for lambda_l1 in [0.0,0.2,0.4,0.6,0.8,1.0]:
            for lambda_l2 in [0.0,0.2,0.4,0.6,0.8,1.0]:
                for min_split_gain in [0.0,1.0]:
                    self.config.params['lambda_l1'] = lambda_l1
                    self.config.params['lambda_l2'] = lambda_l2
                    self.config.params['min_split_gain'] = min_split_gain
                    cv_results = lgb.cv(
                                        self.config.params,
                                        lgbdata,
                                        seed = seed,
                                        nfold = nfold,
                                        early_stopping_rounds = early_stopping_rounds,
                                        verbose_eval = 0
                                        )
                    if self.config.flag == -1:
                        mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).max()
                    else:
                        mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()

                    if mean_merror * self.config.flag < self.config.min_merror * self.config.flag:
                        self.config.min_merror = mean_merror
                        best_params['lambda_l1'] = lambda_l1
                        best_params['lambda_l2'] = lambda_l2
                        best_params['min_split_gain'] = min_split_gain
        self.config.params.update(best_params)
 
    def adj_eta(self, lgbdata, seed = 66, nfold = 5, early_stopping_rounds = 100):
        best_params = {}
        for eta in [0.01, 0.015, 0.025, 0.05, 0.1]:
            self.config.params['learning_rate'] = eta
            cv_results = lgb.cv(
                                self.config.params,
                                lgbdata,
                                seed = seed,
                                nfold = nfold,
                                num_boost_round = self.config.num_boost_round,
                                early_stopping_rounds = early_stopping_rounds,
                                verbose_eval = 0
                                )
            if self.config.flag == -1:
                mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).max()
            else:
                mean_merror = pd.Series(cv_results[self.config.params['metric'] + '-mean']).min()

            if mean_merror * self.config.flag < self.config.min_merror * self.config.flag:
                self.config.min_merror = mean_merror
                best_params['learning_rate'] = eta
        self.config.params.update(best_params)


