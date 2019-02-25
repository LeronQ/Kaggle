
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[144]:


train_data=pd.read_csv('data/breast_cancer_data.csv')
train_data=train_data.drop(['id'],axis=1)


# In[145]:


# 或者采用map： train_data['diagnosis']=train_data['diagnosis'].map({'M':1,'B':0})



train_data[train_data['diagnosis']=='M']=0
train_data[train_data['diagnosis']=='B']=1


# 1：查看数据基本信息

# In[146]:


train_data.info()


# 2：查看数据统计信息

# In[147]:


train_data.describe()


# 3：数据可视化探索

# In[148]:


plt.rc('font',family='SimHei',size=13)

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

plt.subplot2grid((1,1),(0,0))             # 在一张大图里分列几个小图
train_data.diagnosis.value_counts().plot(kind='bar')# 柱状图 
plt.title('好坏情况分布') # 标题
plt.ylabel(u"人数") 
plt.grid()



# fig=plt.figure(figsize=(12,8))

# ax1 = fig.add_subplot(211)

# # plt.add_subplot(211)
# train_data.diagnosis.value_counts().plot(kind='bar')
# ax1.title('人数分布')
# ax1.ylabel('人数')


# In[149]:


# 查看结果好坏比

# good_bad_rate=len(train_data[train_data['diagnosis']=='B']==True)/len(train_data[train_data['diagnosis']=='M']==True)
# good_bad_rate


# 查看特征重要性

# In[152]:


from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(train_data, test_size=0.3, random_state=1)

# import xgboost as xgb

from xgboost import plot_importance

dtrain=xgb.DMatrix(train_dataset.iloc[:,1:],label=train_dataset['diagnosis'])
dtest=xgb.DMatrix(test_dataset.iloc[:,1:],label=test_dataset['diagnosis'])


params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',  # 分类问题
    # 'num_class': 10,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.01,  # 如同学习率
    'seed': 1000,
    'nthread': 7,  # cpu 线程数
    'eval_metric': 'auc'
}

plst = list(params.items())
num_rounds = 500  # 迭代次数
evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 10
bst = xgb.train(plst, dtrain, num_rounds, evallist)

plot_importance(bst)
plt.savefig("importance.png",dpi=120)




# In[ ]:


# 查看特征直接的关联性


# In[159]:


# 程序结果不正常，可能有错误

import seaborn as sns
# corr = train_data.corr()
# xticks = train_data.columns.tolist()
# yticks = list(corr.index)
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1, annot_kws={'size': 5,  'color': 'blue'})
# ax1.set_xticklabels(xticks, rotation=90, fontsize=10)
# ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
# plt.show()




# In[ ]:


prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','area_mean','radius_mean']


train_dataset_X=train_dataset[prediction_var]
train_dataset_y=train_dataset['diagnosis']

test_dataset_X=test_dataset[prediction_var]
test_dataset_y=test_dataset['diagnosis']

del prediction_var


# In[ ]:


# 建立模型预测

# 随机森林分类
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

model=RandomForestClassifier(n_estimators=100)
model.fit(train_dataset_X,train_dataset_y)

prediction=model.predict(test_dataset_X)

metrics.accuracy_score(prediction,test_dataset_y)


# SVM 分类
# from sklearn import svm

# model = svm.SVC()
# model.fit(train_dataset_X,train_dataset_y)
# prediction=model.predict(test_dataset_X)
# metrics.accuracy_score(prediction,test_dataset_y)



resu=model.predict_proba(test_dataset_X)

# 画出 auc曲线 
from sklearn.metrics import roc_curve,auc

fpr,tpr,thershold=roc_curve(test_dataset_y,resu[:,1])
rocauc=auc(fpr,tpr)
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'%rocauc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('随机森林预测')
plt.show()

