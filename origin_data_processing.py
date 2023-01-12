# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:51:05 2021

@author: Administrator
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import decomposition
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd
import calendar
import sklearn.preprocessing as pre_processing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os
import numpy as np
from matplotlib.font_manager import *
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号‘-’显示为方块的问题
sns.set(font='SimHei') #解决Seaborn中文显示问题
#%matplotlib inline

#Step 1:环境准备（导入相关库
## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
#%matplotlib inline

## 数据处理
from sklearn import preprocessing

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

## 模型预测的
import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


#Step 2:数据读取

datasets = pd.read_excel('df_conbine7_9.xlsx')

y_data = datasets['全里程百公里油耗（L）']


#一共是103维度，现在为了设计训练与测试集，将'全里程百公里油耗（L）'作为目标变量提出来
df_cols = datasets.columns
df_cols = list(df_cols)
df_cols.remove('全里程百公里油耗（L）')

## 1) 载入训练集和测试集；
x_data = datasets[df_cols]
y_data = datasets['全里程百公里油耗（L）']
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.25)
## 输出数据的大小信息
print('Train data shape:',x_train.shape)
print('TestA data shape:',y_test.shape)


## 1) 载入训练集和测试集；
#x_data = datasets[df_cols]
#y_data = datasets['全里程百公里油耗（L）']
Train_data,Test_data = train_test_split(datasets,test_size = 0.25)
## 输出数据的大小信息
print('Train data shape:',Train_data.shape)
print('TestA data shape:',Test_data.shape)
#1) 数据简要浏览
## 通过.head() 简要浏览读取数据的形式

print(Train_data.head())

# nan可视化
missing = Train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
plt.figure(figsize=(20,12))
missing.plot.bar()
plt.savefig('./测试集中含nan列的可视化.jpg')

#车牌号为空是缺失数据
print(Train_data['车牌号'].shape) #(205811,)
print(Train_data.loc[Train_data['车牌号'].isnull()])

#抽取出 '车牌号'特征 为 NaN 的样本：
print(Train_data[np.isnan(Train_data['油耗量（当天）'])])


#3) 数据统计信息浏览
#显示所有列
pd.set_option('display.max_columns', None)
## 通过 .describe() 可以查看数值特征列的一些统计信息
print(Train_data.describe())


#抽取出 '车牌号'特征 为 NaN 的样本：
print(Train_data[np.isnan(Train_data['油耗量（当天）'])])

#Step 3: 数据分析（EDA）
#1) 提取数值类型特征列名
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)

#查看非数值的变量名称
categorical_cols = Train_data.select_dtypes(include = 'object').columns
categorical_cols = list(categorical_cols)
print(categorical_cols,'\n')

print(Train_data[numerical_cols].info())

Train_data['date_parsed'] = None
from datetime import datetime
Train_data['date_parsed'] = pd.to_datetime(Train_data['日期y/m/d'],format = '%m%d')
Train_data = Train_data.drop('日期y/m/d',axis = 1)
#修改数据采集时间的格式（训练数据）
Train_data['date_parsed'] = Train_data['date_parsed'].apply(lambda x : x.strftime('2020-%m-%d')).astype('datetime64')
print(Train_data['date_parsed'])

#Train_data['日期y/m/d']
#日期处理好后重新提取数值类型特征列名
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)
print(Train_data[numerical_cols].info())
print(Train_data[categorical_cols].info())

#查看非数值变量里种类
for i in categorical_cols:
    print(i,':',set(Train_data[i]),'\n\n')


#由于数据特征可以分为 三种不同类型的特征，分别为时间特征，类别特征和数值类型特征 , 我们对于不同特征进行分类别处理。
date_features = ['登记日期', 'date_parsed']

categorical_features = [ '设备分类', '任务数据组合异常标记（邮包）', '任务数据组合异常标记（设备）', '任务里程超限异常', '全里程与任务里程组合异常', '里程与时速组合异常', '急刹车次数值异常', '静驶时长值异常',
                        '静驶时长点火占比超限','平均时速值异常','平均时速误差超限','全里程百公里油耗值异常','总油耗误差超限','里程误差超限','异常数据说明','943速递批次',
                       '单位组','急刹车次数是否异常','118重汽','930速递','报废状态','24小时内离线','设备状态','设备类型','能源类型','单位分组','车型','规格型号',
                       '车辆名称','使用单位','配属单位','车牌号']

numeric_features = [] 
for i in categorical_cols:
    if i not in  categorical_features:
        numeric_features.append(i)

#首先对于时间特征进行处理:
from tqdm import tqdm

def date_proc(x):
#    print(type(x))
    m = int(x[5:7])
    if m == 0:
        m = 1
#    print(m)
    return x[:4] + '-' + str(m) + '-' + x[8:]


def num_to_date(df,date_cols):
    for f in tqdm(date_cols):
#        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))   #????出了问题
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return df
Train_data2 = num_to_date(Train_data,date_features)

#首先对于时间特征进行处理:
from tqdm import tqdm

def date_proc(x):
#    print(type(x))
    m = int(x[5:7])
    if m == 0:
        m = 1
#    print(m)
    return x[:4] + '-' + str(m) + '-' + x[8:]


def num_to_date(df,date_cols):
    for f in tqdm(date_cols):
#        df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))   #????出了问题
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
    return df
Test_data2 = num_to_date(Test_data,date_features)


plt.figure()
plt.figure(figsize=(16, 6))
i = 1
for f in date_features:
    for col in ['year', 'month', 'day', 'dayofweek']:
        plt.subplot(2, 4, i)
        i += 1
        v = Train_data[f + '_' + col].value_counts()
        fig = sns.barplot(x=v.index, y=v.values)
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(f + '_' + col)
plt.tight_layout()
plt.show()



plt.figure()
plt.figure(figsize=(16, 6))
i = 1
for f in date_features:
    for col in ['year', 'month', 'day', 'dayofweek']:
        plt.subplot(2, 4, i)
        i += 1
        v = Test_data[f + '_' + col].value_counts()
        fig = sns.barplot(x=v.index, y=v.values)
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(f + '_' + col)
plt.tight_layout()
plt.show()


plt.figure()
plt.figure(figsize=(16, 6))
i = 1
for f in date_features:
    for col in ['year', 'month', 'day', 'dayofweek']:
        plt.subplot(2, 4, i)
        i += 1
        fig = sns.boxplot(x=Train_data[f + '_' + col], y=Train_data['全里程百公里油耗（L）'])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(f + '_' + col)
plt.tight_layout()
plt.show()

#从上面时间分析更新数据特征，'date_parsed_year'没有价值
date_features = ['登记日期_year', '登记日期_month', '登记日期_day',
       '登记日期_dayofweek',  'date_parsed_month',
       'date_parsed_day', 'date_parsed_dayofweek']


#类别特征处理：
#对于类别类型的特征，首先这里第一步进行类别数量统计
from scipy.stats import mode

def sta_cate(df,cols):
    sta_df = pd.DataFrame(columns = ['column','nunique','miss_rate','most_value','most_value_counts','max_value_counts_rate'])
    for col in cols:
        count = df[col].count()
        nunique = df[col].nunique()
        miss_rate = (df.shape[0] - count) / df.shape[0]
        most_value = df[col].value_counts().index[0]
        most_value_counts = df[col].value_counts().values[0]
        max_value_counts_rate = most_value_counts / df.shape[0]
        
        sta_df = sta_df.append({'column':col,'nunique':nunique,'miss_rate':miss_rate,'most_value':most_value,
                                'most_value_counts':most_value_counts,'max_value_counts_rate':max_value_counts_rate},ignore_index=True)
    return sta_df



#训练集类别数量统计
sta_cate(Train_data,categorical_features)

#测试集类别数量统计
sta_cate(Test_data,categorical_features)


#从上述样本的统计情况来看,其中 车牌号 特征特征数量众多，不适宜做类别编码。使用单位 特征需要做进一步的考虑，这里我们先对于 剩余的类别特征做统计可视化：
plt.figure()
plt.figure(figsize=(15, 60))
i = 1
for col in ['设备分类', '任务数据组合异常标记（邮包）','任务数据组合异常标记（设备）', '任务里程超限异常',
 '全里程与任务里程组合异常', '里程与时速组合异常','急刹车次数值异常','静驶时长值异常', '静驶时长点火占比超限',
 '平均时速值异常', '平均时速误差超限', '全里程百公里油耗值异常', '总油耗误差超限',
 '里程误差超限','异常数据说明','943速递批次', '单位组','急刹车次数是否异常','118重汽',
 '930速递', '报废状态', '24小时内离线', '设备状态', '设备类型','能源类型', '单位分组', '车型',
 '车辆名称']:
    plt.subplot(10, 3, i)
    i += 1
    fig = sns.boxplot(x=Train_data[col], y=Train_data['全里程百公里油耗（L）'])
    for item in fig.get_xticklabels():
        item.set_rotation(90)

plt.tight_layout()
plt.show()
plt.savefig('./类别特征做统计可视化1.jpg')


plt.figure()
plt.figure(figsize=(15, 20))
i = 1
for col in [ '规格型号', '配属单位']:
    plt.subplot(2, 1, i)
    i += 1
    fig = sns.boxplot(x=Train_data[col], y=Train_data['全里程百公里油耗（L）'])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
plt.tight_layout()
plt.show()
plt.savefig('./类别特征做统计可视化2（ 规格型号, 配属单位）.jpg')


plt.figure()
plt.figure(figsize=(24, 12))

for col in ['使用单位']:
    plt.subplot(1, 1, 1)
    fig = sns.boxplot(x=Train_data[col], y=Train_data['全里程百公里油耗（L）'])
    for item in fig.get_xticklabels():
        item.set_rotation(90)
plt.tight_layout()
plt.show()
plt.savefig('./类别特征做统计可视化3（使用单位）.jpg')




from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def cate_encoder(df,df_test,cols):
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False,categories ='auto')
    
    for col in cols:
        print(col+':')
        print(set(df[col]))
        print(set(df_test[col]))
        
        le = le.fit(df[col])
        integer_encoded = le.transform(df[col])
        integer_encoded_test = le.transform(df_test[col])

        # binary encode
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
        ohe = ohe.fit(integer_encoded)
        
        onehot_encoded = ohe.transform(integer_encoded)
        onehot_encoded_df = pd.DataFrame(onehot_encoded)
        onehot_encoded_df.columns = list(map(lambda x:str(x)+'_'+col,onehot_encoded_df.columns.values))

        onehot_encoded_test = ohe.transform(integer_encoded_test)
        onehot_encoded_test_df = pd.DataFrame(onehot_encoded_test)
        onehot_encoded_test_df.columns = list(map(lambda x:str(x)+'_'+col,onehot_encoded_test_df.columns.values))
        
        df =  pd.concat([df,onehot_encoded_df], axis=1)
        df_test =  pd.concat([df_test,onehot_encoded_test_df], axis=1)
    
    return df,df_test


cate_cols = ['设备分类', '任务数据组合异常标记（邮包）','任务数据组合异常标记（设备）', '任务里程超限异常',
 '全里程与任务里程组合异常', '里程与时速组合异常','急刹车次数值异常','静驶时长值异常', '静驶时长点火占比超限',
 '平均时速值异常', '平均时速误差超限', '全里程百公里油耗值异常', '总油耗误差超限',
 '里程误差超限','异常数据说明','943速递批次', '单位组','急刹车次数是否异常','118重汽',
 '930速递', '报废状态', '24小时内离线', '设备状态', '设备类型','能源类型', '单位分组', '车型',
 '车辆名称']
Train_data[cate_cols] = str(Train_data[cate_cols].fillna(-1))
Test_data[cate_cols] = str(Test_data[cate_cols].fillna(-1))

## 对类别特征进行 OneEncoder
# data = pd.get_dummies(data, columns=cate_cols)

Train_data,Test_data = cate_encoder(Train_data,Test_data,cate_cols)


