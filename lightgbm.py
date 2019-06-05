## =========== import 必要的模块 =========== ##
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# 读取数据
# 这里使用原始特征+tf_idf特征
dpath = './data/'

train1 = pd.read_csv(dpath +"Otto_FE_train_org.csv")
train2 = pd.read_csv(dpath +"Otto_FE_train_tfidf.csv")

# 去掉多余的id
train2 = train2.drop(["id","target"], axis=1)
train =  pd.concat([train1, train2], axis = 1, ignore_index=False)

del train1
del train2

# 准备数据
# 将类别字符串变成数字，LightGBM不支持字符串格式的特征输入/标签输入
y_train = train['target'] # 形式为Class_x
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s) - 1) # 将类别的形式由Class_x变为0-8之间的整数

X_train = train.drop(["id", "target"], axis=1)

# 保存特征名字以备后用（可视化）
feat_names = X_train.columns 

# sklearn的学习器大多支持稀疏数据输入，模型训练会快很多
# 查看一个学习器是否支持稀疏数据，可以看fit函数是否支持: X: {array-like, sparse matrix}.
# 可自行用timeit比较稠密数据和稀疏数据的训练时间
from scipy.sparse import csr_matrix
X_train = csr_matrix(X_train)

## ========== LightGBM超参数调优 ========= ##

'''
LightGBM的主要的超参包括：
1.树的数目n_estimators 和 学习率 learning_rate
2.树的最大深度max_depth 和 树的最大叶子节点数目num_leaves
（注意：XGBoost只有max_depth，LightGBM采用叶子优先的方式生成树，num_leaves很重要，设置成比 2^max_depth 小）
3.叶子结点的最小样本数:min_data_in_leaf(min_data, min_child_samples)
4.每棵树的列采样比例：feature_fraction/colsample_bytree
5.每棵树的行采样比例：bagging_fraction （需同时设置bagging_freq=1）/subsample
6.正则化参数lambda_l1(reg_alpha), lambda_l2(reg_lambda)
'''

MAX_ROUNDS = 10000
# 相同的交叉验证分组
# prepare cross validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)

# 1. n_estimators
# 直接调用lightgbm内嵌的交叉验证(cv)，可对连续的n_estimators参数进行快速交叉验证
# 而GridSearchCV只能对有限个参数进行交叉验证，且速度相对较慢
def get_n_estimators(params , X_train , y_train , early_stopping_rounds=10):
    lgbm_params = params.copy()
    lgbm_params['num_class'] = 9
     
    lgbmtrain = lgbm.Dataset(X_train , y_train )
     
    # num_boost_round为弱分类器数目，下面的代码参数里因为已经设置了early_stopping_rounds
    # 即性能未提升的次数超过过早停止设置的数值，则停止训练
    cv_result = lgbm.cv(lgbm_params , lgbmtrain , num_boost_round=MAX_ROUNDS , nfold=3,  metrics='multi_logloss' , early_stopping_rounds=early_stopping_rounds,seed=3 )
     
    print('best n_estimators:' , len(cv_result['multi_logloss-mean']))
    print('best cv score:' , cv_result['multi_logloss-mean'][-1])
     
    return len(cv_result['multi_logloss-mean'])
    
    
 params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'n_jobs': 4,
          'learning_rate': 0.1,
          'num_leaves': 60,
          'max_depth': 6,
          'max_bin': 127,   # 2^6,原始特征为整数，很少超过100
          'subsample': 0.7,
          'bagging_freq': 1,
          'colsample_bytree': 0.7,
         }

n_estimators_1 = get_n_estimators(params , X_train , y_train)
# 运行结果
# best n_estimators: 428
# best cv score: 0.47523858964819493

# 2. num_leaves & max_depth=7
# num_leaves建议70-80，搜索区间50-80,值越大模型越复杂，越容易过拟合 相应的扩大max_depth=7

params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'num_class':9, 
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'max_depth': 7,
          'max_bin': 127,   # 2^6,原始特征为整数，很少超过100
          'subsample': 0.7,
          'bagging_freq': 1,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

num_leaves_s = range(50,90,10) #50,60,70,80
tuned_parameters = dict( num_leaves = num_leaves_s)

grid_search = GridSearchCV(lg, n_jobs=4, param_grid=tuned_parameters, cv = kfold, scoring="neg_log_loss", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_

# examine the best model
print(-grid_search.best_score_)
print(grid_search.best_params_)

# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

n_leafs = len(num_leaves_s)

x_axis = num_leaves_s
plt.plot(x_axis, -test_means)
#plt.errorbar(x_axis, -test_means, yerr=test_stds,label = ' Test')
#plt.errorbar(x_axis, -train_means, yerr=train_stds,label = ' Train')
plt.xlabel( 'num_leaves' )
plt.ylabel( 'Log Loss' )
plt.show()

# 3. min_child_samples
params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'num_class':9, 
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'max_depth': 7,
          'num_leaves':70,
          'max_bin': 127, #2^6,原始特征为整数，很少超过100
          'subsample': 0.7,
          'bagging_freq': 1,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

min_child_samples_s = range(10,50,10) 
tuned_parameters = dict( min_child_samples = min_child_samples_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="neg_log_loss", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
grid_search.best_estimator_

# examine the best model
print(-grid_search.best_score_)
print(grid_search.best_params_)

# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = min_child_samples_s

plt.plot(x_axis, -test_means)
#plt.errorbar(x_axis, -test_scores, yerr=test_stds ,label = ' Test')
#plt.errorbar(x_axis, -train_scores, yerr=train_stds,label =  +' Train')

plt.show()

min_child_samples=30

# 行采样参数 sub_samples/bagging_fraction

params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'num_class':9, 
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'max_depth': 7,
          'num_leaves':70,
          'min_child_samples':30,
          'max_bin': 127,  # 2^6,原始特征为整数，很少超过100
          #'subsample': 0.7,
          'bagging_freq': 1,
          'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

subsample_s = [i/10.0 for i in range(5,10)]
tuned_parameters = dict( subsample = subsample_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="neg_log_loss", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_

# examine the best model
print(-grid_search.best_score_)
print(grid_search.best_params_)

# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = subsample_s

plt.plot(x_axis, -test_means)
#plt.errorbar(x_axis, -test_scores[:,i], yerr=test_stds[:,i] ,label = str(max_depths[i]) +' Test')
#plt.errorbar(x_axis, -train_scores[:,i], yerr=train_stds[:,i] ,label = str(max_depths[i]) +' Train')

plt.show()

# 列采样参数 sub_feature/feature_fraction/colsample_bytree
params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'num_class':9, 
          'n_jobs': 4,
          'learning_rate': 0.1,
          'n_estimators':n_estimators_1,
          'max_depth': 7,
          'num_leaves':70,
          'min_child_samples':30,
          'max_bin': 127, #2^6,原始特征为整数，很少超过100
          'subsample': 0.8,
          'bagging_freq': 1,
          #'colsample_bytree': 0.7,
         }
lg = LGBMClassifier(silent=False,  **params)

colsample_bytree_s = [i/10.0 for i in range(5,10)]
tuned_parameters = dict( colsample_bytree = colsample_bytree_s)

grid_search = GridSearchCV(lg, n_jobs=4,  param_grid=tuned_parameters, cv = kfold, scoring="neg_log_loss", verbose=5, refit = False)
grid_search.fit(X_train , y_train)
#grid_search.best_estimator_

# examine the best model
print(-grid_search.best_score_)
print(grid_search.best_params_)

# plot CV误差曲线
test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = colsample_bytree_s

plt.plot(x_axis, -test_means)
#plt.errorbar(x_axis, -test_scores[:,i], yerr=test_stds[:,i] ,label = str(max_depths[i]) +' Test')
#plt.errorbar(x_axis, -train_scores[:,i], yerr=train_stds[:,i] ,label = str(max_depths[i]) +' Train')

plt.show()

# 用所有训练数据，采用最佳参数重新训练模型
params = {'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'num_class':9, 
          'n_jobs': 4,
          'learning_rate': 0.01,
          'n_estimators':n_estimators_2,
          'max_depth': 7,
          'num_leaves':75,
          'min_child_samples':40,
          'max_bin': 127, #2^6,原始特征为整数，很少超过100
          'subsample': 0.8,
          'bagging_freq': 1,
          'colsample_bytree': 0.4,
         }
lg = LGBMClassifier(silent=False,  **params)
lg.fit(X_train, y_train)

# 保存模型，用于后续测试
import pickle

pickle.dump(lg, open("Otto_LightGBM_org_tfidf.pkl", 'wb'))

# 特征重要性
df = pd.DataFrame({"columns":list(feat_names), "importance":list(lg.feature_importances_.T)})
df = df.sort_values(by=['importance'],ascending=False)
df
