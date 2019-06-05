## ==================== import 必要的模块 ============= ##
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

# 读取数据
dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")

### ==================== 特征工程 ==================== ###

# 分开特征和标签
y_train = train['target'] 
# 暂存id
train_id = train['id']
# drop ids and get labels
X_train = train.drop(["id", "target"], axis=1)

# 保存特征名字
columns_org = X_train.columns

# feat编码：log(x+1)
X_train_log = np.log1p(X_train)

# 重新组成DataFrame
feat_names = columns_org + "_log"
X_train_log = pd.DataFrame(columns = feat_names, data = X_train_log.values)

# feat编码：TF-IDF
# transform counts to TFIDF features
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

# 输出稀疏矩阵
X_train_tfidf = tfidf.fit_transform(X_train).toarray()

# 重新组成DataFrame,为了可视化
feat_names = columns_org + "_tfidf"
X_train_tfidf = pd.DataFrame(columns = feat_names, data = X_train_tfidf)

# 数据预处理
# 对原始数据缩放
from sklearn.preprocessing import MinMaxScaler
# 构造输入特征的标准化器
ms_org = MinMaxScaler()

# 保存特征名字，用于结果保存为csv
feat_names_org = X_train.columns

# 用训练训练模型（得到均值和标准差）：fit
# 并对训练数据进行特征缩放：transform
X_train = ms_org.fit_transform(X_train)

# 对log数据缩放
# 构造输入特征的标准化器
ms_log = MinMaxScaler()

# 保存特征名字，用于结果保存为csv
feat_names_log = X_train_log.columns

# 用训练训练模型（得到均值和标准差）：fit
# 并对训练数据进行特征缩放：transform
X_train_log = ms_log.fit_transform(X_train_log)


# 对tf-idf数据缩放
# 构造输入特征的标准化器
ms_tfidf = MinMaxScaler()

# 保存特征名字，用于结果保存为csv
feat_names_tfidf = X_train_tfidf.columns

# 用训练训练模型（得到均值和标准差）：fit
# 并对训练数据进行特征缩放：transform
X_train_tfidf = ms_tfidf.fit_transform(X_train_tfidf)

# 保存原始特征
y = pd.Series(data = y_train, name = 'target')
feat_names = columns_org
train_org = pd.concat([train_id, pd.DataFrame(columns = feat_names_org, data = X_train), y], axis = 1)
train_org.to_csv(dpath +'Otto_FE_train_org.csv',index=False,header=True)

# 保存log特征变换结果
y = pd.Series(data = y_train, name = 'target')
train_log = pd.concat([train_id, pd.DataFrame(columns = feat_names_log, data = X_train_log), y], axis = 1)
train_log.to_csv(dpath +'Otto_FE_train_log.csv',index=False,header=True)

# 保存tf-idf特征变换结果
y = pd.Series(data = y_train, name = 'target')
train_tfidf = pd.concat([train_id, pd.DataFrame(columns = feat_names_tfidf, data = X_train_tfidf), y], axis = 1)
train_tfidf.to_csv(dpath +'Otto_FE_train_tfidf.csv',index=False,header=True)

# 保存特征编码过程中用到的模型，用于后续对测试数据的特征编码
import pickle

pickle.dump(tfidf, open("tfidf.pkl", 'wb'))
pickle.dump(ms_org, open("MinMaxSclaer_org.pkl", 'wb'))
pickle.dump(ms_log, open("MinMaxSclaer_log.pkl", 'wb'))
pickle.dump(ms_tfidf, open("MinMaxSclaer_tfidf.pkl", 'wb'))
