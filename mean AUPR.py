import os
import time
import numpy as np
import pandas as pd
import csv
import math
import random
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)  # 生成列的横坐标，横坐标都是x0，纵坐标变化
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return
    # 画虚线框
    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    # 连接两个虚线框
    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    # 小虚框内各点坐标
    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    # 画虚线框内的点
    # 把小虚框内的任一点减去小虚框左下角点生成相对坐标，再乘以倍数（4）加大虚框左下角点
    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return

def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # 分母可能出现0，需要讨论待续
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result

i = 0
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'bisque', 'sienna']

# 用于保存混淆矩阵
AllResult = []
Ps = []
Rs = []
RPs = []
mean_R = np.linspace(0, 1, 1000)


#DW
counter0 = 0
print(i)
Ps1 = []
Rs1 = []
RPs1 = []
mean_R1 = np.linspace(0, 1, 1000)
while counter0 < 5:
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'RF2_RealAndPredictionA+B'+ str(counter0) + '.csv'
    RAPNameProb = 'RF2_RealAndPredictionProbA+B'+ str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps1.append(interp(mean_R1, precision, recall))
    RPs1.append(average_precision)

    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P1 = np.mean(Ps1, axis=0)
print(mean_P1)
mean_RPs1 = np.mean(RPs1, axis=0)
print(mean_RPs1)
std_RPs = np.std(RPs1)
plt.plot(mean_P1, mean_R1, color='red',
         label=r'RF (AUPR = %0.4f)' % (mean_RPs1),
         lw=2, alpha=0.8)

#------Line------
counter0 = 0
print(i)
Ps2 = []
Rs2 = []
RPs2 = []
mean_R2 = np.linspace(0, 1, 1000)
while counter0 < 5:
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'SVM2_RealAndPredictionA+B'+ str(counter0) + '.csv'
    RAPNameProb = 'SVM2_RealAndPredictionProbA+B'+ str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps2.append(interp(mean_R2, precision, recall))
    RPs2.append(average_precision)
    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P2 = np.mean(Ps2, axis=0)
print(mean_P2)
mean_RPs2 = np.mean(RPs2, axis=0)
print(mean_RPs2)
std_RPs = np.std(RPs2)
plt.plot(mean_P2, mean_R2, color='gold',
         label=r'SVM (AUPR = %0.4f)' % (mean_RPs2),
         lw=2, alpha=0.8)

#hope
counter0 = 0
print(i)
Ps3 = []
Rs3 = []
RPs3 = []
mean_R3 = np.linspace(0, 1, 1000)
while counter0 < 5:
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'KNN2_RealAndPredictionA+B'+ str(counter0) + '.csv'
    RAPNameProb = 'KNN2_RealAndPredictionProbA+B'+ str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps3.append(interp(mean_R3, precision, recall))
    RPs3.append(average_precision)
    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P3 = np.mean(Ps3, axis=0)
print(mean_P3)
mean_RPs3 = np.mean(RPs3, axis=0)
print(mean_RPs3)
std_RPs = np.std(RPs3)
plt.plot(mean_P3, mean_R3, color='purple',
         label=r'KNN (AUPR = %0.4f)' % (mean_RPs3),
         lw=2, alpha=0.8)

#LR
counter0 = 0
Ps4 = []
Rs4 = []
RPs4 = []
mean_R4 = np.linspace(0, 1, 1000)
while counter0 < 5:
    print(i)
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'GBD2_RealAndPredictionA+B'+ str(counter0) + '.csv'
    RAPNameProb = 'GBD2_RealAndPredictionProbA+B'+ str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps4.append(interp(mean_R4, precision, recall))
    RPs4.append(average_precision)
    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P4 = np.mean(Ps4, axis=0)
print(mean_P4)
mean_RPs4 = np.mean(RPs4, axis=0)
print(mean_RPs4)
std_RPs4 = np.std(RPs4)
plt.plot(mean_P4, mean_R4, color='sienna',
         label=r'GBDT (AUPR = %0.4f)' % (mean_RPs4),
         lw=2, alpha=0.8)


#SMmiR3
counter0 = 0
Ps5 = []
Rs5 = []
RPs5 = []
mean_R5 = np.linspace(0, 1, 1000)
while counter0 < 5:
    print(i)
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'lineRealAndPredictionA+B' + str(counter0) + '.csv'
    RAPNameProb = 'lineRealAndPredictionProbA+B' + str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps5.append(interp(mean_R5, precision, recall))
    RPs5.append(average_precision)

    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P5 = np.mean(Ps5, axis=0)
print(mean_P5)
mean_RPs5 = np.mean(RPs5, axis=0)
print(mean_RPs5)
std_RPs = np.std(RPs5)
plt.plot(mean_P5, mean_R5, color='green',
         label=r'Line (AUPR = %0.4f)' % (mean_RPs5),
         lw=2, alpha=0.8)

#SMmiR3
counter0 = 0
Ps6 = []
Rs6 = []
RPs6 = []
mean_R6 = np.linspace(0, 1, 1000)
while counter0 < 5:
    print(i)
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = 'deepwalkRealAndPredictionA+B' + str(counter0) + '.csv'
    RAPNameProb = 'deepwalkRealAndPredictionProbA+B' + str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps6.append(interp(mean_R6, precision, recall))
    RPs6.append(average_precision)

    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P6 = np.mean(Ps6, axis=0)
print(mean_P6)
mean_RPs6 = np.mean(RPs6, axis=0)
print(mean_RPs6)
std_RPs6 = np.std(RPs6)
plt.plot(mean_P6, mean_R6, color='blue',
         label=r'DeepWalk (AUPR = %0.4f)' % (mean_RPs6),
         lw=2, alpha=0.8)




#SMmiR3
counter0 = 0
Ps7 = []
Rs7 = []
RPs7 = []
mean_R7 = np.linspace(0, 1, 1000)
while counter0 < 5:
    print(i)
    # 读取文件
    RealAndPrediction = []
    RealAndPredictionProb = []
    RAPName = '64RealAndPredictionA+B' + str(counter0) + '.csv'
    RAPNameProb = '64RealAndPredictionProbA+B' + str(counter0) + '.csv'
    ReadMyCsv(RealAndPrediction, RAPName)
    ReadMyCsv(RealAndPredictionProb, RAPNameProb)
    # 生成Real和Prediction
    Real = []
    Prediction_ncDR = []
    PredictionProb_ncDR = []
    counter = 0
    while counter < len(RealAndPrediction):
        Real.append(int(RealAndPrediction[counter][0]))
        Prediction_ncDR.append(RealAndPrediction[counter][1])
        PredictionProb_ncDR.append(RealAndPredictionProb[counter][1])
        counter = counter + 1

    average_precision = average_precision_score(Real, PredictionProb_ncDR)
    precision, recall, _ = precision_recall_curve(Real, PredictionProb_ncDR)

    Ps7.append(interp(mean_R7, precision, recall))
    RPs7.append(average_precision)

    i += 1
    counter0 = counter0 + 1
# # 画均值
mean_P7 = np.mean(Ps7, axis=0)
print(mean_P7)
mean_RPs7 = np.mean(RPs7, axis=0)
print(mean_RPs7)
std_RPs = np.std(RPs7)
plt.plot(mean_P7, mean_R7, color='black',
         label=r'PTBGRP (AUPR = %0.4f)' % (mean_RPs7),
         lw=2, alpha=0.8)
#-------------------------------------------------------------------------------------------------#
# # 大画均值
# Ps.append(mean_P1)
# Ps.append(mean_P2)
# Ps.append(mean_P3)
# Ps.append(mean_P4)
# Ps.append(mean_P5)
# RPs.append(mean_RPs1)
# RPs.append(mean_RPs2)
# RPs.append(mean_RPs3)
# RPs.append(mean_RPs4)
# RPs.append(mean_RPs5)
# mean_Pa = np.mean(Ps, axis=0)
# print(mean_P1)
# mean_RPsa = np.mean(RPs, axis=0)
# print(mean_RPsa)
# std_RPs = np.std(RPs)
# MyEnlarge(0, 0.3, 0.75, 0.75, 0.5, 0, 2, mean_Pa, mean_R, 2, colorlist[5])


#画表头，标签
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.ylim([-0.02, 1.02])
plt.xlim([-0.02, 1.02])
plt.title('Precision-Recall curves')
# 画网格
plt.grid(linestyle='--')
# 画对角线
plt.plot([1, 0], [0, 1], color='navy', lw=2, alpha=0.8, linestyle='--')
plt.legend(bbox_to_anchor=(0.54, 0.44))

plt.savefig('mean_PR-5fold.svg', dpi=300)
plt.savefig('mean_PR-5fold.pdf', dpi=300)
plt.savefig('mean_PR-5fold.tif', dpi=300)
plt.show()























