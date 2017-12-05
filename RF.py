import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier as rf
from matplotlib import pyplot as plt
from sklearn import metrics

# Read in our input data
train = pd.read_csv('../input/train.csv')
submit = pd.read_csv('../input/test.csv')

# This prints out (rows, columns) in each dataframe
print('Train shape:', train.shape)
print('Test shape:', submit.shape)

y = train.target.values
id_test = submit['id'].values

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

# Create an XGBoost-compatible metric from Gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# feature engineering

train['ps_reg_01'] = train['ps_reg_01'] * 10
train['ps_reg_01'].astype(int)

train['ps_reg_02'] = train['ps_reg_02'] * 10
train['ps_reg_02'].astype(int)

submit['ps_reg_01'] = submit['ps_reg_01'] * 10
submit['ps_reg_01'].astype(int)

submit['ps_reg_02'] = submit['ps_reg_02'] * 10
submit['ps_reg_02'].astype(int)

train['ps_ind_02_cat'].astype('category')
train['ps_ind_04_cat'].astype('category')
train['ps_ind_05_cat'].astype('category')

submit['ps_ind_02_cat'].astype('category')
submit['ps_ind_04_cat'].astype('category')
submit['ps_ind_05_cat'].astype('category')

train['ps_car_01_cat'].astype('category')
train['ps_car_02_cat'].astype('category')
train['ps_car_03_cat'].astype('category')
train['ps_car_04_cat'].astype('category')
train['ps_car_05_cat'].astype('category')
train['ps_car_06_cat'].astype('category')
train['ps_car_07_cat'].astype('category')
train['ps_car_08_cat'].astype('category')
train['ps_car_09_cat'].astype('category')
train['ps_car_10_cat'].astype('category')
train['ps_car_11_cat'].astype('category')

submit['ps_car_01_cat'].astype('category')
submit['ps_car_02_cat'].astype('category')
submit['ps_car_03_cat'].astype('category')
submit['ps_car_04_cat'].astype('category')
submit['ps_car_05_cat'].astype('category')
submit['ps_car_06_cat'].astype('category')
submit['ps_car_07_cat'].astype('category')
submit['ps_car_08_cat'].astype('category')
submit['ps_car_09_cat'].astype('category')
submit['ps_car_10_cat'].astype('category')
submit['ps_car_11_cat'].astype('category')

for i in range(9):
    cat_name = 'ps_car_0' + str(i+1) + '_cat'
    cat_name_bayes = 'ps_car_0' + str(i+1) + '_cat_bayes'
    n_df = train.groupby(cat_name, as_index = False)['target'].mean()
    n_df[cat_name_bayes] = n_df['target']
    del n_df['target']
    train = pd.merge(train,n_df,how='left', on=cat_name)
    submit = pd.merge(submit,n_df,how='left', on=cat_name)

n_df = train.groupby('ps_car_10_cat', as_index = False)['target'].mean()
n_df['ps_car_10_cat_bayes'] = n_df['target']
del n_df['target']
train = pd.merge(train,n_df,how='left', on='ps_car_10_cat')
submit = pd.merge(submit,n_df,how='left', on='ps_car_10_cat')

n_df = train.groupby('ps_car_11_cat', as_index = False)['target'].mean()
n_df['ps_car_11_cat_bayes'] = n_df['target']
del n_df['target']
train = pd.merge(train,n_df,how='left', on='ps_car_11_cat')
submit = pd.merge(submit,n_df,how='left', on='ps_car_11_cat')

train['nan_sum'] = (train == -1).sum(axis= 1)
submit['nan_sum'] = (submit == -1).sum(axis= 1)

float_feature = ['ps_reg_01','ps_reg_02','ps_reg_03','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_14','ps_ind_15']
for item in float_feature:
    for j in range(float_feature.index(item)+1,len(float_feature)):
        new_name = item +' * '+ float_feature[j]
        train[new_name] = train[item] * train[float_feature[j]]
        submit[new_name] = submit[item] * submit[float_feature[j]]
        # new_name = item + ' / ' + float_feature[j]
        # train[new_name] = train[item] / train[float_feature[j]]
        # submit[new_name] = submit[item] / submit[float_feature[j]]

# df=pd.read_csv('../script/feature importance.csv')
# df=df.sort_values(by='importance',ascending=False)
# feature_all=np.array(df['feature'])
# print feature_all.shape
# df=df[df['importance']>0.001]
# feature_want=np.array(df['feature'])
# feature_unwant = [item for item in feature_all if item not in feature_want]
# train = train.drop(feature_unwant, axis=1)
# submit = submit.drop(feature_unwant, axis=1)


unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)
submit = submit.drop(unwanted, axis=1)

# ending
train = train.drop(['id', 'target'], axis=1)
submit = submit.drop(['id'], axis=1)
feature_names = [item for item in train.columns]



# Create a submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = np.zeros_like(id_test)

X = train.values
kfold = 3
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=1234)
gini_valid_res = []

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    sss_valid = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=5678)

    for j, (train_index, valid_index) in enumerate(sss_valid.split(X_train, y_train)):

        print('[Fold %d/%d, tour %d/%d]' % (i + 1, kfold, j+1, kfold))

        X_train_without_val, X_valid = X_train[train_index], X_train[valid_index]
        y_train_without_val, y_valid = y_train[train_index], y_train[valid_index]

        print X_train_without_val.shape
        print X_valid.shape
        print X_test.shape
        print submit.shape

        # Convert our data into LGBoost format
        # d_train = rf.Dataset(X_train_without_val, y_train_without_val)
        # d_valid = rf.Dataset(X_valid, y_valid)
        # if feature_names: d_train.set_feature_name(feature_name=feature_names)

        # d_train.set_categorical_feature(['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat'])
        # submit.set_categorical_feature(['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_c at'])
        d_train=X_train
        d_valid=X_valid


        d_submit = submit.values
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        # Train the model! We pass in a max of 2,000 rounds (with early stopping after 100)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        rf_clf=rf(n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,
                  min_samples_leaf=1, min_weight_fraction_leaf=0, max_features='auto')
        rf_clf.fit(X_train,y_train)
        # validation score for normalized gini coefficient
        y_test_predicted = rf_clf.predict_proba(X_test)
        gini_valid_res.append(gini_normalized(y_test,y_test_predicted))

        # Predict on our test data
        p_submit = rf_clf.predict_proba(d_submit) # n class in column

        sub['target'] += p_submit[:,1] / (kfold*kfold)

# Create a submission file
sub.to_csv('result.csv', index=False)

# show result for gini_score for validation set
print "result this turn : "
print np.mean(gini_valid_res)

