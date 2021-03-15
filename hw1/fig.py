import matplotlib.pyplot as plt
from numpy import *
from lr import *

def plotBestFit(weights, weight_record):  #画出最终分类的图
    plt.switch_backend('agg')
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # x = arange(-3.0, 3.0, 0.1)
    x_min = min(dataArr[:,1])
    x_max = max(dataArr[:,1])
    weights = weights.getA()
    x = arange(x_min, x_max, 1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print("Drawing")
    ax.plot(x, y, '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('./figure/LR_GA_1000000.jpg')
    # plt.show()
    
    fig_weight = plt.figure()
    cycles, w_num, one = shape(weight_record)
    weight_0 = zeros(cycles)
    weight_1 = zeros(cycles)
    weight_2 = zeros(cycles)
    ax_x = arange(1,cycles+1,1)
    for i in range(cycles):
        weight_0[i] = weight_record[i][0][0]
        weight_1[i] = weight_record[i][1][0]
        weight_2[i] = weight_record[i][2][0]
    weight_0.reshape((cycles,))
    # print(weight_0)
    weight_1.reshape((cycles,))
    # print(weight_1)
    weight_2.reshape((cycles,))
    # print(weight_2)
    # Weight 0
    w_0 = fig_weight.add_subplot(3,1,1)
    w_0.plot(ax_x, weight_0, '-g')
    plt.ylabel('W0')
    # Weight 1
    w_1 = fig_weight.add_subplot(3,1,2)
    w_1.plot(ax_x, weight_1, '-b')
    plt.ylabel('W1')
    # Weight 2
    w_1 = fig_weight.add_subplot(3,1,3)
    w_1.plot(ax_x, weight_2, '-r')
    plt.ylabel('W2')
    
    plt.savefig('./figure/Weights_GA_1000000.jpg')
    # plt.show()