import matplotlib.pyplot as plt
import numpy as np
from PIL.EpsImagePlugin import split
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# 定义函数
def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):       # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return
def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


def GenerateFeature(pairs,encode1,encode2):
    sampel_feature = []
    print("nihao",len(pairs))
    for i in range(len(pairs)):
        drug = pairs[i][0]
        mirna = pairs[i][1]
        print(len(encode1))
        print(len(encode2))
        temp = []
        for a in range(len(AllDrugCanonicalSMILES)):
            if drug == AllDrugCanonicalSMILES[a][0]:
                temp.extend(encode1[a][1:])
                print(temp)
        for b in range(len(AllMiSequence)):
            if mirna == AllMiSequence[b][0]:
                temp.extend(encode2[b][1:])
        sampel_feature.append(temp)
    return sampel_feature
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):   # 产生所有miRNA/drug的embedding表示
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        PairFeature.append(SequenceList[counter][0])        # 加入名称

        FeatureMatrix = []
        counter1 = 0            # 生成特征矩阵
        while counter1 < PaddingLength:  # 截取长度
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:  # embedding长度
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass
        ##将特征矩阵64*64=64*0=reshape（0*64）
        result_info = np.array(FeatureMatrix)
        irisPca = PCA(n_components=1)
        pcaDate = irisPca.fit_transform(result_info)
        Date = pcaDate.ravel().reshape(1,64)
        #print('输后矩阵', Date.shape)
        #print('输入后矩阵', Date.tolist())
        PairFeature.append(Date[0].tolist()) #数组转化为列表
        #print(PairFeature)
        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature

def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2):
    SampleFeature1 = []
    SampleFeature2 = []

    counter = 0
    while counter < len(InteractionList):
        Pair1 = InteractionList[counter][0]
        Pair2 = InteractionList[counter][1]

        counter1 = 0
        while counter1 < len(EmbeddingFeature1):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(EmbeddingFeature2):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
            counter2 = counter2 + 1

        counter = counter + 1
    SampleFeature1, SampleFeature2 = np.array(SampleFeature1), np.array(SampleFeature2)
    SampleFeature1.astype('float32')
    SampleFeature2.astype('float32')
    print('1SampleFeature1',np.array(SampleFeature1).shape)
    print('2SampleFeature2',np.array(SampleFeature2).shape)
    #SampleFeature1 = SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[0], SampleFeature1.shape[2], 0)
    #SampleFeature2 = SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[0], SampleFeature2.shape[2], 0)
    return SampleFeature1, SampleFeature2

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break

        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')
    return SampleFeature1, SampleFeature2

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='y',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

def TestOutput(classifier, name):
    # 输出预测值
    ModelTestOutput = classifier.predict_proba(X_test)
    ModelTestOutput1 = classifier.predict(X_test)
    LabelPredictionProb = []
    LabelPrediction = []

    counter = 0
    CounterT = 4
    while counter < len(np.array(y_test_Pre)):
        rowProb = []
        rowProb.append(y_test_Pre[counter])
        rowProb.append(ModelTestOutput[counter][1])
        LabelPredictionProb.append(rowProb)

        row = []
        row.append(y_test_Pre[counter])
        if ModelTestOutput[counter][1] > 0.5:
            row.append(1)
        else:
            row.append(0)
        LabelPrediction.append(row)

        counter = counter + 1
    LabelPredictionProbName = str(name) + 'RealAndPredictionProbA+B' + str(CounterT) + '.csv'
    StorFile(LabelPredictionProb, LabelPredictionProbName)
    LabelPredictionName = str(name) + 'RealAndPredictionA+B' + str(CounterT) + '.csv'
    StorFile(LabelPrediction, LabelPredictionName)

if __name__ == '__main__':

    AllMiSequence = []
    ReadMyCsv1(AllMiSequence, 'phage host sequence988.csv')
    miRNAEmbedding = []
    ReadMyCsv3(miRNAEmbedding, 'phagehost988_Embedding988.csv')

    # AllDrugCanonicalSMILES = []
    # ReadMyCsv1(AllDrugCanonicalSMILES, 'phage sequence.csv')
    # DrugEmbedding = []
    # ReadMyCsv3(DrugEmbedding, 'phage_all_Embedding64.csv')


    #AllCircRNASequence = []
    #AllDrugCanonicalSMILES是一个列表，没有shape属性
    '''
    for c in range(len(AllDrugCanonicalSMILES)):
        m = []
        m = AllDrugCanonicalSMILES[c][0][0:24]
        AllDrugCanonicalSMILES[c][0] = m
    '''
    AllNodeBehavior = []
    ReadMyCsv1(AllNodeBehavior, 'phagehost_Behavior988.csv')

# ------第一折----
    PositiveSample_Train = []
    ReadMyCsv1(PositiveSample_Train, 'PositiveSample_Train_all_phagehost9885.csv')
    PositiveSample_Validation = []
    ReadMyCsv1(PositiveSample_Validation, 'PositiveSample_Validation_all_phagehost9885.csv')
    PositiveSample_Test = []
    ReadMyCsv1(PositiveSample_Test, 'PositiveSample_Test_all_phagehost9885.csv')

    NegativeSample_Train = []
    ReadMyCsv1(NegativeSample_Train, 'NegativeSample_Train_all_phagehost9885.csv')
    NegativeSample_Validation = []
    ReadMyCsv1(NegativeSample_Validation, 'NegativeSample_Validation_all_phagehost9885.csv')
    NegativeSample_Test = []
    ReadMyCsv1(NegativeSample_Test, 'NegativeSample_Test_all_phagehost9885.csv')




 # 下面为circRNA与mirna对的内容
    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)  # ？？？？？
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(PositiveSample_Validation)
    x_test_pair.extend(NegativeSample_Test)
    x_test_pair.extend(NegativeSample_Validation)

    # Label->one hot 真实的，已经被预测过的
    y_train_Pre = MyLabel(x_train_pair)
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)

    # 图嵌入，把对应的向量导入进来
    # DrugEmbeddingFeature = GenerateEmbeddingFeature(AllDrugCanonicalSMILES, DrugEmbedding, 64)  #
    miRNAEmbeddingFeature = GenerateEmbeddingFeature(AllMiSequence, miRNAEmbedding, 64) #

    # # word2vec的特征向量、行为向量
    x_train_1_Attribute, x_train_2_Attribute = GenerateSampleFeature(x_train_pair, miRNAEmbeddingFeature, miRNAEmbeddingFeature)  # drug and miRNA feature. matrix and vector
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateSampleFeature(x_validation_pair, miRNAEmbeddingFeature, miRNAEmbeddingFeature)
    x_test_1_Attribute, x_test_2_Attribute = GenerateSampleFeature(x_test_pair, miRNAEmbeddingFeature, miRNAEmbeddingFeature)


    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)
    x_test_1_Behavior, x_test_2_Behavior = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior)
    print('x_train_1_Attribute','x_train_2_Attribute','x_train_1_Behavior','x_train_2_Behavior',np.array(x_train_1_Attribute).shape,np.array(x_train_2_Attribute).shape,x_train_1_Behavior.shape,x_train_2_Behavior.shape)


    print(x_train_1_Attribute.shape)
    print(x_train_1_Behavior.shape)
    print(x_train_2_Attribute.shape)
    print(x_train_2_Behavior.dtype)
    X_train = []
    X_test = []
    X_validation = []
    for d in range(len(x_train_pair)):
        X_train = np.concatenate((x_train_1_Attribute, x_train_2_Attribute, x_train_1_Behavior, x_train_2_Behavior), axis=1)
    for d in range(len(x_train_pair)):
        X_test = np.concatenate((x_test_1_Attribute, x_test_2_Attribute, x_test_1_Behavior, x_test_2_Behavior), axis=1)


# -----随机森林-------
#     forest = RandomForestClassifier(n_estimators=7, max_features="auto")
#     forest.fit(X_train, np.array(y_train_Pre))
#     prediction = forest.score(X_test,np.array(y_test_Pre))
#     print('RF2_clf_Score', prediction)
#     print('RF2_clf_Proba', forest.predict_proba(X_test))
# # 输出预测值
#     TestOutput(forest, 'RF2_')




# #--------逻辑回归---------
# #     LR_clf = LogisticRegression(penalty='l2',  max_iter=10, random_state=2, multi_class='ovr', solver="liblinear",C=1)#创建分类器
#     LR_clf = LogisticRegression(penalty='l1',tol=0.0001, C=1.0, intercept_scaling=1,
#                                 random_state=None, solver='saga', max_iter=100, multi_class='ovr',
#                                          verbose=0,  n_jobs=1)
#     LR_clf.fit(X_train, np.array(y_train_Pre))#用训练数据拟合分类器模型
#     prediction = LR_clf.score(X_test,np.array(y_test_Pre))
#     print('LR2_clf_Score', prediction)
#     print('LR2_clf_Proba', LR_clf.predict_proba(X_test))
# # 输出预测值
#     TestOutput(LR_clf, 'LR2_')




# # #-----SVM分类-------
#     SVM_clf = svm.SVC(probability = True, C=0.45) #probability = True必须加上这句，否则报错
#     SVM_clf.fit(X_train, np.array(y_train_Pre))
#     prediction = SVM_clf.score(X_test, np.array(y_test_Pre))
#     print('SVM2_clf_Score', prediction)
#     print('SVM2_clf_Proba', SVM_clf.predict_proba(X_test))
# # # 输出预测值
#     TestOutput(SVM_clf, 'SVM2_')





# #KNN分类
#     KNN_clf = KNeighborsClassifier(n_neighbors=2)
#     KNN_clf.fit(X_train,np.array(y_train_Pre))
#     prediction = KNN_clf.score(X_test, np.array(y_test_Pre))
#     print('KNN2_clf_Score', prediction)
#     print('KNN2_clf_Proba', KNN_clf.predict_proba(X_test))
# # # 输出预测值
#     TestOutput(KNN_clf, 'KNN2_')



# #AdaBoost Classifier
#     AdaB_clf = AdaBoostClassifier(learning_rate=100, n_estimators=100)
#     AdaB_clf.fit(X_train, np.array(y_train_Pre))
#     prediction = AdaB_clf.score(X_test, np.array(y_test_Pre))
#     print('AdaB2_clf_Score', prediction)
#     print('AdaB2_clf_Proba', AdaB_clf.predict_proba(X_test))
# # # 输出预测值
#     TestOutput(AdaB_clf, 'AdaB2_')

# # # GBDT(Gradient Boosting Decision Tree) Classifier
    GBDT_clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=5)
    GBDT_clf.fit(X_train,np.array(y_train_Pre))
    prediction = GBDT_clf.score(X_test, np.array(y_test_Pre))
    print('GBDT2_clf_Score', prediction)
    print('GBDT2_clf_Proba', GBDT_clf.predict_proba(X_test))

# # # 输出预测值
    TestOutput(GBDT_clf, 'GBD2_')




