import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb
from matplotlib import pyplot as plt
import json
import graphviz

# Read in our input data
train = pd.read_csv('../input/train.csv')
submit = pd.read_csv('../input//test.csv')

# This prints out (rows, columns) in each dataframe
print('Train shape:', train.shape)
print('Test shape:', submit.shape)

id_test = submit['id'].values
id_train = train['id'].values
y = train['target'].copy()
y_valid_pred = 0 * y

def ginic(actual, pred):
    actual = np.asarray(actual)  # In case, someone passes Series or list
    n = len(actual)
    n = float(n)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n

def gini_normalized(a, p):
    if p.ndim == 2:  # Required for sklearn wrapper
        p = p[:, 1]  # If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,
                  val_series=None,
                  tst_series=None,
                  sub_series=None,
                  target=None,
                  min_samples_leaf=100,
                  smoothing=10,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name +'_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index

    ft_val_series = pd.merge(
        val_series.to_frame(val_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=val_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_val_series.index = val_series.index

    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    ft_sub_series = pd.merge(
        sub_series.to_frame(sub_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=sub_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_sub_series.index = sub_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_val_series, noise_level), add_noise(ft_tst_series, noise_level), add_noise(ft_sub_series, noise_level)



# # # feature selection
# print 'feature selection'
# #
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 20,
    # 'metric': {'AUC'},
    'max_depth':5,
    'min_data_in_leaf':500,
    'learning_rate': 0.02,
    'lambda_l1': 10,
    'lambda_l2': 2,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_split_gain' : 5,
    # 'verbose': 0,
    # 'is_unbalance' : True
}
#
outfile = open('parameter.json','a')
json.dump(params,outfile,ensure_ascii=False)
outfile.write('\n')
train = train.drop(['id', 'target'], axis=1)
submit = submit.drop(['id'], axis=1)
#
# d_train = lgb.Dataset(train.values[:500000], y[:500000])
# d_valid = lgb.Dataset(train.values[500000:], y[500000:])
# print train.values[:500000].shape
# print train.values[500000:].shape
# print y[:500000].shape
#
# watchlist = [(d_train, 'train'), (d_valid, 'valid')]
# feature_names = [item for item in train.columns]
# if feature_names: d_train.set_feature_name(feature_name=feature_names)
# mdl = lgb.train(params, d_train, valid_sets=d_valid, num_boost_round=3000, early_stopping_rounds=5)
# print 'Plot feature importances...'
# ax = lgb.plot_importance(mdl, max_num_features=100)
# plt.show()
#
# importance = mdl.feature_importance(importance_type='gain')
# print importance
#
# importance = importance > 0


# importance = [i for i in range(len(importance)) if importance[i]]
# unwanted = train.columns[list(set([i for i in range(len(train.columns))]) - set(importance))]
# train = train.drop(unwanted, axis=1)
# submit = submit.drop(unwanted, axis=1)

# trainning
# Set xgb parameters

print 'training'

# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = np.zeros_like(id_test)

kfold = 3
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=789)
gini_valid_res = []

for i, (train_index, test_index) in enumerate(sss.split(train.values, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_test_up = train.iloc[train_index,:].copy(), train.iloc[test_index,:].copy()
    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()

    sss_valid = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=456)

    for j, (train_index, valid_index) in enumerate(sss_valid.split(X_train.values, y_train)):

        print('[Fold %d/%d, tour %d/%d]' % (i + 1, kfold, j+1, kfold))

        X_train_without_val, X_valid = X_train.iloc[train_index,:].copy(), X_train.iloc[valid_index,:].copy()
        y_train_without_val, y_valid = y_train.iloc[train_index].copy(), y_train.iloc[valid_index].copy()
        X_test = X_test_up.copy()
        X_submit = submit.copy()

        print X_train_without_val.shape,X_valid.shape,X_test.shape


        # feature engineering
        # feature engineering
        print 'feature engineering'

        X_train_without_val['ps_reg_01'] = X_train_without_val['ps_reg_01'] * 10
        X_train_without_val['ps_reg_01'].astype(int)

        X_train_without_val['ps_reg_02'] = X_train_without_val['ps_reg_02'] * 10
        X_train_without_val['ps_reg_02'].astype(int)

        X_valid['ps_reg_01'] = X_valid['ps_reg_01'] * 10
        X_valid['ps_reg_01'].astype(int)

        X_valid['ps_reg_02'] = X_valid['ps_reg_02'] * 10
        X_valid['ps_reg_02'].astype(int)

        X_test['ps_reg_01'] = X_test['ps_reg_01'] * 10
        X_test['ps_reg_01'].astype(int)

        X_test['ps_reg_02'] = X_test['ps_reg_02'] * 10
        X_test['ps_reg_02'].astype(int)

        X_submit['ps_reg_01'] = X_submit['ps_reg_01'] * 10
        X_submit['ps_reg_01'].astype(int)

        X_submit['ps_reg_02'] = X_submit['ps_reg_02'] * 10
        X_submit['ps_reg_02'].astype(int)

        f_cats = [f for f in X_train_without_val.columns if "_cat" in f]

        for f in f_cats:
            X_train_without_val[f + "_avg"],X_valid[f + "_avg"], X_test[f + "_avg"],X_submit[f + "_avg"] = \
                target_encode(trn_series=X_train_without_val[f],
                            val_series=X_valid[f],
                            tst_series=X_test[f],
                            sub_series=X_submit[f],
                            target=y_train_without_val,
                            min_samples_leaf=500,
                            smoothing=1,
                            noise_level=0)

        X_train_without_val['nan_sum'] = (X_train_without_val == -1).sum(axis=1)
        X_valid['nan_sum'] = (X_valid == -1).sum(axis=1)
        X_test['nan_sum'] = (X_test == -1).sum(axis=1)
        X_submit['nan_sum'] = (X_submit == -1).sum(axis=1)

        float_feature = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15',
                         'ps_calc_01', 'ps_calc_02', 'ps_calc_03']
        for item in float_feature:
            for j in range(float_feature.index(item) + 1, len(float_feature)):
                new_name = item + ' * ' + float_feature[j]
                X_train_without_val[new_name] = X_train_without_val[item] * X_train_without_val[float_feature[j]]
                X_valid[new_name] = X_valid[item] * X_valid[float_feature[j]]
                X_test[new_name] = X_test[item] * X_test[float_feature[j]]
                X_submit[new_name] = X_submit[item] * X_submit[float_feature[j]]

        combs = [
            ('ps_reg_01', 'ps_car_02_cat'),
            ('ps_reg_01', 'ps_car_04_cat'),
        ]
        for n_c, (f1, f2) in enumerate(combs):
            name1 = f1 + "_plus_" + f2
            X_train_without_val[name1] = X_train_without_val[f1].apply(lambda x: str(x)) + "_" + X_train_without_val[f2].apply(lambda x: str(x))
            X_valid[name1] = X_valid[f1].apply(lambda x: str(x)) + "_" + X_valid[f2].apply(lambda x: str(x))
            X_test[name1] = X_test[f1].apply(lambda x: str(x)) + "_" + X_test[f2].apply(lambda x: str(x))
            X_submit[name1] = X_submit[f1].apply(lambda x: str(x)) + "_" + X_submit[f2].apply(lambda x: str(x))

            # Label Encode
            lbl = LabelEncoder()
            lbl.fit(list(X_train_without_val[name1].values) + list(X_valid[name1].values) + list(X_test[name1].values) + list(X_submit[name1].values))
            X_train_without_val[name1] = lbl.transform(list(X_train_without_val[name1].values))
            X_valid[name1] = lbl.transform(list(X_valid[name1].values))
            X_test[name1] = lbl.transform(list(X_test[name1].values))
            X_submit[name1] = lbl.transform(list(X_submit[name1].values))

        unwanted = X_train_without_val.columns[X_train_without_val.columns.str.startswith('ps_calc_')]
        X_train_without_val = X_train_without_val.drop(unwanted, axis=1)
        X_valid = X_valid.drop(unwanted, axis=1)
        X_test = X_test.drop(unwanted, axis=1)
        X_submit = X_submit.drop(unwanted, axis=1)

        print X_train_without_val.shape,X_valid.shape,X_test.shape

        # Convert our data into LGBoost format
        d_train = lgb.Dataset(X_train_without_val, y_train_without_val)
        d_valid = lgb.Dataset(X_valid, y_valid)

        # ending
        feature_names = [item for item in X_train_without_val.columns]
        if feature_names: d_train.set_feature_name(feature_name=feature_names)

        d_submit = X_submit.values
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        # Train the model! We pass in a max of 2,000 rounds (with early stopping after 100)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        mdl = lgb.train(params, d_train, valid_sets=d_valid, num_boost_round=3000, early_stopping_rounds=5)

        # validation score for normeliezd gini coefficient
        y_test_predicted = mdl.predict(X_test)
        print gini_normalized(y_test,y_test_predicted)
        gini_valid_res.append(gini_normalized(y_test,y_test_predicted))

        y_valid_pred.iloc[test_index] += y_test_predicted/3

        # Predict on our test data
        p_submit = mdl.predict(d_submit)
        sub['target'] += p_submit / (kfold*kfold)

# Create a stacking file
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('../result/lgbm.csv', float_format='%.6f', index=False)

# Create a submission file
sub.to_csv('../result/submit_lgb.csv', index=False)

# show result for gini_score for validation set
print "result this turn : "
print np.mean(gini_valid_res)
outfile.write('local_score: '+str(np.mean(gini_valid_res)))
outfile.write('\n')
outfile.write('**************************************')
outfile.write('\n')
outfile.close()