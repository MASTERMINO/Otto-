## ========== import 必要的模块 ========== ##
import pandas as pd 
import numpy as np

# 读取数据
dpath = './data/'
test1 = pd.read_csv(dpath +"Otto_FE_test_org.csv")
test2 = pd.read_csv(dpath +"Otto_FE_test_tfidf.csv")

# 去掉多余的id
test2 = test2.drop(["id"], axis=1)
test =  pd.concat([test1, test2], axis = 1, ignore_index=False)

# 准备数据
test_id = test['id']   
X_test = test.drop(["id"], axis=1)

# 保存特征名字以备后用（可视化）
feat_names = X_test.columns 

from scipy.sparse import csr_matrix
X_test = csr_matrix(X_test)

# load训练好的模型
import pickle
model = pickle.load(open("Otto_LightGBM_org_tfidf.pkl", 'rb'))

# 输出每类的概率
y_test_pred = model.predict_proba(X_test)

# 输出每类的概率
y_test_pred = model.predict_proba(X_test)

y_test_pred.shape

# 生成提交结果
out_df = pd.DataFrame(y_test_pred)

columns = np.empty(9, dtype=object)
for i in range(9):
    columns[i] = 'Class_' + str(i+1)

out_df.columns = columns

out_df = pd.concat([test_id,out_df], axis = 1)
out_df.to_csv("LightGBM_org_tfidf.csv", index=False)
