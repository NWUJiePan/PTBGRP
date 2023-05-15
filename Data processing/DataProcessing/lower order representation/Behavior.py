from numpy import *
import numpy as np
import random
import math
import os
import time
import pandas as pd
import csv
import math
import random
import copy

# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


if __name__ == '__main__':

    # AllDrug = []
    # ReadMyCsv(AllDrug, 'RNAInterAllDrug.csv')
    # AllMi = []
    # ReadMyCsv(AllMi, 'RNAInterAllMi.csv')
    #
    # AllNode = []
    # AllNode.extend(AllDrug)
    # AllNode.extend(AllMi)

    AllNode = []
    ReadMyCsv(AllNode, 'phage host list .csv')

    print(len(AllNode))

    NodeEmbeddingName = 'phage_host_nodehope64.txt'
    NodeEmbedding = np.loadtxt(NodeEmbeddingName, dtype=str, skiprows=1)


    # Behavior
    AllNodeBehavior = []
    counter = 0
    while counter < len(AllNode):
        pair = []
        pair.append(AllNode[counter][0])
        counter1 = 0
        while counter1 < len(NodeEmbedding[0]) - 1:  # 如果节点孤立，则Feature全为0，与embedding统一维度
            pair.append(0)
            counter1 = counter1 + 1
        AllNodeBehavior.append(pair)
        counter = counter + 1

    print(np.array(AllNodeBehavior).shape)

    counter = 0
    while counter < len(NodeEmbedding):
        counter1 = 0
        while counter1 < len(AllNode):
            if AllNode[counter1][0] == NodeEmbedding[counter][0]:
                break
            counter1 = counter1 + 1

        # print('counter', counter)
        # print('counter1', counter1)

        AllNodeBehavior[counter1][1:] = NodeEmbedding[counter][1:]
        counter = counter + 1

    print(np.array(AllNodeBehavior).shape)

    StorFile(AllNodeBehavior, 'phagehost_Behaviorhope64.csv')

