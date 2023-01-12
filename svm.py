# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:58:37 2023

@author: Xiongqi
"""

# import numpy as np
# import pylab as plt
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
import random

def model(x_train,x_test,y_train,y_test):#模型主体，方便更换x与y
    model = SVR(gamma='auto'); 
    print(model)
    model.fit(x_train,y_train); 
    pred_y_train = model.predict(x_train)
    R2_train_all=r2_score(y_train,pred_y_train)
    EV_train_all=explained_variance_score(y_train,pred_y_train)
    MSE_train_all=mean_squared_error(y_train, pred_y_train)
    pred_y_test = model.predict(x_test)
    print("原始数据与预测值前15个值对比：")
    for i in range(15): print(y_test[i],pred_y_test[i])
    R2_test_all=r2_score(y_test,pred_y_test)
    EV_test_all=explained_variance_score(y_test,pred_y_test)
    MSE_test_all=mean_squared_error(y_test, pred_y_test)
    print("训练集R2:",R2_train_all)
    print("测试集R2：", R2_test_all)
    print("训练集EV:",EV_train_all)
    print("测试集EV：", EV_test_all)
    print("训练集MSE:",MSE_train_all)
    print("测试集MSE：", MSE_test_all)
    return(pred_y_test,R2_test_all,R2_train_all,EV_test_all,EV_train_all,MSE_test_all,MSE_train_all)


#定义训练集测试集划分函数
def train_test_group(x,y,test_size):
    x_index=list(x.index)#构建X的横坐标列表
    random.shuffle(x_index)#打乱X_index
    test_index=x_index[0:int(len(x_index)*test_size)]#取出打乱之后的前面百分之20的样本作为测试集
    train_index=x_index[int(len(x_index)*test_size):len(x_index)]#取出后面的百分之80作为训练集
    x_test=x.loc[test_index]#取出x的测试集
    x_train=x.loc[train_index]#取出x的训练集
    y_test=y[test_index]#取出y的测试集
    y_train=y[train_index]#取出y的训练集
    return x_train,x_test,y_train,y_test


random.seed(1)#设置随机种子固定分组
#使用玉米基因组SNP数据预测玉米产量
data_path='E:/桌面/机器学习导引/ori_data_csv/'
x_ori_1=pd.read_csv(data_path+'genotype.csv',sep=',',index_col=0)
y_ori_1=pd.read_csv(data_path+'RIL-Phenotypes.csv',index_col=0)
y_1=y_ori_1['yd']
x_1=x_ori_1
x_train_1,x_test_1,y_train_1,y_test_1 = train_test_group(x_1, y_1, 0.2)#将所有样本中的百分之80作为训练集
#将训练集与测试集的样本名记录下来
train_index=list(x_train_1.index)
test_index=list(x_test_1.index)

#将训练集与测试集标签写入文件
f=open(data_path+"train_index.csv","w")
f.write('\n'.join(train_index))
f.close()
f=open(data_path+"test_index.csv","w")
f.write('\n'.join(test_index))
f.close()

#随机分组先生成行等量的数字，打乱之后按照数字将行号取出来再进行分组
x_index_1=list(x_1.index)
random.shuffle(x_index_1)
pred_y_test_1,R2_test_all_1,R2_train_all_1,EV_test_all_1,EV_train_all_1,MSE_test_all_1,MSE_train_all_1= model(x_train_1,x_test_1,y_train_1,y_test_1)

#使用玉米转录组数据预测玉米产量
x_ori_2=pd.read_csv(data_path+'genotype.csv',sep=',',index_col=0)
y_ori_2=pd.read_csv(data_path+'RIL-Phenotypes.csv',index_col=0)
y_2=y_ori_2['yd'].values
x_2=x_ori_2.values
x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(x_2, y_2, test_size=0.2)#将所有样本中的百分之80作为训练集
pred_y_test_2,R2_test_all_2,R2_train_all_2,EV_test_all_2,EV_train_all_2,MSE_test_all_2,MSE_train_all_2= model(x_train_2,x_test_2,y_train_2,y_test_2)

#使用玉米代谢组数据预测玉米产量
x_ori_3=pd.read_csv(data_path+'genotype.csv',sep=',',index_col=0)
y_ori_3=pd.read_csv(data_path+'RIL-Phenotypes.csv',index_col=0)
y_3=y_ori_3['yd'].values
x_3=x_ori_3.values
x_train_3,x_test_3,y_train_3,y_test_3 = train_test_split(x_3, y_3, test_size=0.2)#将所有样本中的百分之80作为训练集
pred_y_test_3,R2_test_all_3,R2_train_all_3,EV_test_all_3,EV_train_all_3,MSE_test_all_3,MSE_train_all_3= model(x_train_3,x_test_3,y_train_3,y_test_3)

#绘制柱形图
