# 首先 import 必要的模块
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

读取数据

dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")
train.head()
