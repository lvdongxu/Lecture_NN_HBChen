# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:45:12 2021

@author: lvdongxu
"""

from numpy import *
import os
import argparse

parser = argparse.ArgumentParser(description="Import the parameters of the LR")
parser.add_argument('--pre_or_not', '-p', choices=[0,1], default=0, type=int, help='Preprocessing or not')
parser.add_argument('--ascent_way', '-a', choices=[0,1], default=0, type=int, help='Choosing the Gradient Ascent way')
args=parser.parse_args()

filename='./data.txt' #文件目录
pre_or_not = args.pre_or_not #选择是否进行数据集的预处理, 1:进行预处理; 0:不进行预处理
ascent_way = args.ascent_way #选择梯度上升的方式，1: Stochastic Gradient Ascent; 0: Gradient Ascent


def loadDataSet(pre_or_not):   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        # lineArr = line.strip().split()
        lineArr = line.split(",")
        # print(lineArr[0], lineArr[1], lineArr[2])
        lineArr_0 = float(lineArr[0]) if pre_or_not == 0 else float(lineArr[0]) / 10
        lineArr_1 = float(lineArr[1]) if pre_or_not == 0 else float(lineArr[1]) / 10
        dataMat.append([1.0, lineArr_0, lineArr_1])   #前面的1，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(X):  #sigmoid函数
    return 1.0/(1+exp(-X))

def predict(weights, dataMatrix):
    '''
    计算得到的是等于1的概率，如果>=0.5，则认为得到的预测结果为1，否则为0
    weights   : 权重值，W1+W2*X1+W3*X2
    dataMatrix: 数据，X1,X2
    '''
    preLabel = sigmoid(dataMatrix * weights)
    preProb  = preLabel
    for label_i in preLabel:
        if label_i >= 0.5:
            label_i = 1
        else:
            label_i = 0
    return preLabel, preProb 

def cost_function(preProb, labelMat):
    '''
    计算Loss
    preProb: 预测得到的分类概率结果
    labelMat: 数据集自带的标签值
    '''
    m = shape(preProb)[0]
    loss = 0.0
    for i in range(m):
        if preProb[i] >= 0 and preProb[i] <= 1:
            loss -= labelMat[i] * log(preProb[i]) + (1 - labelMat[i]) * log(preProb[i])
        else:
            loss -= 0.0
    return loss / m

def predict_error_rate(preLabel, LabelMat):
    '''
    计算预测的准确率
    preLabel: 预测得到的分类标签结果
    labelMat: 数据集自带的标签值
    '''
    m = shape(preLabel)[0]
    errRate = 0.0
    for i in range(m):
        if int(preLabel[i]) == int(LabelMat[i]):
            errRate += 1
        else:
            errRate += 0
    return errRate / m

def gradAscent(dataMat, labelMat, alpha, maxCycles): #梯度上升求最优参数
    '''
    Gradient Ascent梯度上升函数
    dataMat: list, 数据集的数据记录
    labelMat: list, 数据集的标签记录
    '''

    dataMatrix     = mat(dataMat)              #将读取的数据转换为矩阵
    classLabels    = mat(labelMat).transpose() #将读取的数据转换为矩阵
    m, n           = shape(dataMatrix)
    weights        = ones((n,1))               #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示三个参数。
    weight_record  = zeros((maxCycles,n,1))    #权重记录，用于展示每次训练得到的权重变化 
    loss_record    = zeros((maxCycles,1))      #loss记录画图 
    errRate_record = zeros((maxCycles,1))      #Error Rate记录画图 

    for k in range(maxCycles):
        print("Gradient Ascent -- 第",k,"次迭代")
        h                 = sigmoid(dataMatrix * weights)
        error             = (classLabels - h)                                 #求导后差值
        weights           = weights + alpha * dataMatrix.transpose() * error  #迭代更新权重
        preLabel, preProb = predict(weights, dataMatrix)
        loss              = cost_function(preProb, classLabels)
        errRate           = predict_error_rate(preLabel, classLabels)
        weight_record[k]  = weights
        loss_record[k]    = loss
        errRate_record[k] = errRate
    return weights, weight_record, loss_record, errRate_record

def stocGradAscent0(dataMat, labelMat, alpha, maxCycles):  
    '''
    随机梯度上升，当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。所以采用每次迭代中一次只选择其中的一行数据进行更新权重。
    dataMat: list, 数据集的数据记录
    labelMat: list, 数据集的标签记录
    '''
    
    dataMatrix     = mat(dataMat)
    classLabels    = labelMat
    m,n            = shape(dataMatrix)
    weights        = ones((n,1))
    weight_record  = ones((maxCycles,n,1)) #权重记录，用于展示每次训练得到的权重变化 
    loss_record    = zeros((maxCycles,1)) #loss记录画图 
    errRate_record = zeros((maxCycles,1)) #Error Rate记录画图 

    for k in range(maxCycles):
        print("Stochastic Gradient Ascent v0 -- 第",k,"次迭代")
        for i in range(m): #遍历计算每一行
            h       = sigmoid(sum(dataMatrix[i] * weights))
            error   = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i].transpose()
        preLabel, preProb = predict(weights, dataMatrix)
        loss              = cost_function(preProb, classLabels)
        errRate           = predict_error_rate(preLabel, classLabels)
        weight_record[k]  = weights
        loss_record[k]    = loss
        errRate_record[k] = errRate
    return weights, weight_record, loss_record, errRate_record

# def stocGradAscent1(dataMat, labelMat): #改进版随机梯度上升，在每次迭代中随机选择样本来更新权重，并且随迭代次数增加，权重变化越小。
#     dataMatrix=mat(dataMat)
#     classLabels=labelMat
#     m,n=shape(dataMatrix)
#     weights=ones((n,1))
#     maxCycles=1000000
#     weight_record = ones((maxCycles,n,1)) #权重记录，用于展示每次训练得到的权重变化 
#     for j in range(maxCycles): #迭代
#         print("Stochastic Gradient Ascent v1 -- 第",j,"次迭代")
#         dataIndex=[i for i in range(m)]
#         for i in range(m): #随机遍历每一行
#             alpha=4/(1+j+i)+0.0001  #随迭代次数增加，权重变化越小。
#             randIndex=int(random.uniform(0,len(dataIndex)))  #随机抽样
#             h=sigmoid(sum(dataMatrix[randIndex]*weights))
#             error=classLabels[randIndex]-h
#             weights=weights+alpha*error*dataMatrix[randIndex].transpose()
#             del(dataIndex[randIndex]) #去除已经抽取的样本
#         weight_record[j] = weights
#     return weights
    
# def main():
#     dataMat, labelMat = loadDataSet()
#     weights, weight_record = gradAscent(dataMat, labelMat).getA()
#     # weights = stocGradAscent0(dataMat, labelMat).getA()
#     plotBestFit(weights, weight_record)

if __name__=='__main__':
    # main()
    dataMat, labelMat = loadDataSet(pre_or_not)
    alpha             = 0.001
    maxCycles         = 100000 if pre_or_not == 1 else 1000000    
    npy_name_1        = "pre"  if pre_or_not == 1 else "no_pre"
    npy_name_2        = "Grad" if ascent_way == 1 else "stocGrad"
    if ascent_way == 0:
        weights, weight_record, loss_record, errRate_record = gradAscent(dataMat, labelMat, alpha, maxCycles)
    else:
        weights, weight_record, loss_record, errRate_record = stocGradAscent0(dataMat, labelMat, alpha, maxCycles)
    
    weights_name        = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'weights.npy'
    weight_record_name  = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'weight_record.npy'
    loss_record_name    = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'loss_record.npy'
    errRate_record_name = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'errRate_record.npy'

    save(weights_name,        weights       )
    save(weight_record_name,  weight_record )
    save(loss_record_name,    loss_record   )
    save(errRate_record_name, errRate_record)

    txt_weights_name        = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'weights.txt'
    txt_weight_record_name  = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'weight_record.txt'
    txt_loss_record_name    = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'loss_record.txt'
    txt_errRate_record_name = './data/' + npy_name_1 + '_' + npy_name_2 + '_' + 'errRate_record.txt'

    savetxt(txt_weights_name,        weights       )
    savetxt(txt_weight_record_name,  weight_record )
    savetxt(txt_loss_record_name,    loss_record   )
    savetxt(txt_errRate_record_name, errRate_record)
