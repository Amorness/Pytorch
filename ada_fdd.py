# coding:utf-8

"""
Created on Tue Jan 16 16:23:57 2019

@author: zhaoyingying

算法一：
     输入：预处理之后的风机数据
     输出：发电机轴承温度故障的风机，及对应的概率
     步骤：
         1、为每台风机找邻居, 有两种找邻居的方法 a) 通过负载（即有功功率的相似性）；
                                           b) 为风机找邻居，根据系数（a,b）,采用autoGMM算法。该系数的含义是风机自身的运行状况
                                              其中（a,b）来自于 前轴承温度= a * 有功功率 + b * 机舱内温度 + c
            在杨家湾电站上采用b）方法为风机找邻居

         2、对每台风机建模，建模数据 从  自身数据和邻居风机的数据中选取，选取评分最高的模型；
         3、为每台风机预测残差；
         4、根据互为邻居的风机数据计算其是否为故障。

"""
"""
Modified on Tue Jan 30 20:46:57 2019
@author: zhaoyingying

测试：
1，如何降低false alarm? done
2, 需要的最短的数据时长是多少？在杨家湾电站上目前需要3个月的数据3-month
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.neural_network import MLPRegressor

from utils.clustering import fdGMM
from utils.path_set import directory
from matplotlib import pyplot as plt
from utils.date_tool import getDayList
import datetime


def cluster_wts(dataPath, MAX_NEIGHBORS=2, FEA=['TR002']):
    """
    通过找相似的负载（有功功率 TR002），找一台风机的邻居

    Parameters
    -------
        dataPath: string. 预处理之后的数据
        MAX_NEIGHBORS: int. 指定一台风机最大的邻居数
        FEA: list.  TR002. 有功功率

    Returns
    -------
        neighborsDict. key: 一台风机的device code
                       value: 所有邻居
        WTs: list,该电站上所有风机的device code
    """
    # 合并所有风机的有功功率，并滤波
    flist = glob.glob(dataPath + '/*.csv')
    print("flist==", flist)
    WTs = []
    medfiltLen = 600  #
    diffDf = pd.DataFrame()
    for idx, f in enumerate(flist):
        df = pd.read_csv(f, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.loc[:, FEA]
        wtName = os.path.basename(f)[0:-4]
        df = filter_df(df, medfiltLen)  # filtering the data
        WTs.append(wtName)  # adding to wind turbine list
        diffDf.loc[:, wtName] = df.loc[:, FEA[0]]

    # 计算风机之间两两的pearson相似性
    corr = diffDf.corr()

    # 根据最大的邻居数，找最相似的邻居
    neighborsDict = dict.fromkeys(WTs)
    for idx, wt in enumerate(WTs):
        tmp_df = corr.loc[:, [wt]]
        tmp_df = tmp_df.sort_values(by=wt, ascending=False)
        neighbors = tmp_df.index.tolist()[0:MAX_NEIGHBORS]
        neighborsDict[wt] = neighbors
    print('Successfully find neighbors for all turbines!')
    return neighborsDict, WTs


def cluster_wts_improved(dataPath, FEA=['TR002', 'NC005', 'GN010']):
    """
    根据发电机前轴承温度和有功功率、机舱内环境温度的关系（alpha 和beta）对风机进行聚类。

    Parameters
    -------
        dataPath: string. 预处理之后的数据路径
        FEA: list. ['TR002','NC001','GN010']
            TR002: 有功功率
            NC005: 机舱内温度
            GN010: 发电机前轴承温度
            x:['TR002','NC001']
            y: ['GN010']

    Returns
    -------
        neighborsDict: dictionary  (key, value)
                        key: 风机的device code
                       value: 该风机对应的邻居list
        WTs: list. 所有风机的device code组成的list
    """
    flist = glob.glob(dataPath + '/*.csv')
    WTs = []  # 所有风机的device code
    equationList = []  # 发电机前轴承温度和有功功率、机舱内环境温度的关系（alpha 和beta）
    #medfiltLen = 72  #残差滤波的长度 only test by zyy

    #为每台风机找发电机前轴承温度和有功功率、机舱内环境温度的关系
    for idx, f in enumerate(flist):
        print(f)
        df = pd.read_csv(f, index_col=0)  #设置时间列为索引，并指定为日期格式
        df.index = pd.DatetimeIndex(df.index)
        df.index =  [i.replace(tzinfo= None) for i in df.index]
        df = df[FEA]  #只获取发电机前轴承温度、舱内温度、有功功率三列属性
        wtName = os.path.basename(f)[0:-4]  #获取风机的名字，去掉后缀(.csv)
        #df = filter_df(df, medfiltLen) only test by zyy
        df = df.apply(lambda x: (x - np.mean(x)) / np.std(x))  #归一化

        x = df[FEA[0:-1]]  #定义输入为：有功功率和舱内温度
        y = df[FEA[-1:]]  # 定义输出为：前轴承温度

        # 获取输入和输出之间的线性关系，用神经网络多层感知器 - 回归模型实现
        model_mlp1 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3,), activation='identity', max_iter=200,
                                  random_state=1)
        model_mlp1.fit(x, y)  #预测输入和输出之间的关系
        coef = model_mlp1.coefs_
        inter = model_mlp1.intercepts_
        equation = ann_single_equation(x, coef, inter)  #用公式描述关系，例如：y= alpha * x1+ beta * x2 + gama
        equation = equation[0:-1]  # 得到alpha 和 beta
        equationList.append(equation)
        WTs.append(wtName)

    # 对（alpha， beta）进行聚类，用 autoGMM 方法
    dimension = len(equationList[0])
    maxClusterNum = len(equationList)
    fK, means_re, labels, abnormal_proba, cova_re = fdGMM(equationList, dimension, maxClusterNum)

    clusterNum = len(set(labels))  #风机聚类后类的数量
    print('Sucessfully find ' + str(clusterNum) + ' clusters for this site!')

    #以字典的形式保存风机和对应的邻居
    clusterDict = dict()
    clusterDf = pd.DataFrame(index=WTs, columns=['label'], data=labels)
    for idx, wt in enumerate(WTs):
        lb = labels[idx]
        clusterDict[wt] = clusterDf[clusterDf.label == lb].index.tolist()

    return clusterDict, WTs


def merge_neighbors(wtName, wtNeighbors, dataPath, enable = False):
    """
    把一台风机所有的邻居数据合并在一起

    Parameters
    -------
        wtName: string. 一台指定风机的device code
        wtNeighbors: dictionary. (key, value). key: 一台风机的device code. value: 该风机所有邻居风机的devicecode
        dataPath: string. 要合并的数据路径，在这里是预处理之后的数据

    Returns
    -------
        df: datafram. 以追加（append）的方式合并数据到同一个dataframe中。保证合并的文件和合并前的文件有同样的列和索引（index）
    """
    #读取待合并的风机数据，设置索引为日期时间
    df = pd.read_csv(dataPath + '/' + wtName + '.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index = [i.replace(tzinfo = None) for i in df.index]
    #df = df[(df.index >= startDate) & (df.index <= endDate)] only test for zyy

    #如果强制设置为不合并，则返回合并前的数据
    if enable is False:
        return df

    # 合并所有的邻居
    for wt in wtNeighbors:
        if wt != wtName:  #把所有非自身的邻居以追加的方式合并
            tmpDf = pd.read_csv(dataPath + '/' + wt + '.csv', index_col=0)
            tmpDf.index = pd.to_datetime(tmpDf.index)
            df = df.append(tmpDf)
    return df

def calculate_residuals(df, wt, wtPath):
    """
    对每台风机，计算残差。残差= 预测的前后轴承温差 - 实际的前后轴承温差
    Parameters
    -------
        df: dataframe. The merged data that contains all neighbors for a turbine
        wt: string. a turbine's device code
        wtPath: the preprocessed data path for a wind trubine
    return:
        df: dataframe. the residuals. index is datetime. columns is the residuals for a turbine
    """

    inputFea = ['TR002', 'NC005', 'GN013']  #分别对应有功功率、机舱温度、u相温度
    outputFea1 = 'GN010'  # 发电机前轴承温度
    outputFea2 = 'GN011'  # 发电机后轴承温度

    #归一化
    data = df.apply(lambda x: (x - np.mean(x)) / np.std(x))
    mn = df.mean().tolist()
    std = df.std().tolist()

    #如果没有u相温度，则移除之
    if 'GN013' not in data.columns.tolist():
        inputFea.remove('GN013')
        print('This site DONOT collect generators U temperature. That may effect the performance of FDD model.')

    #定义输入、输出
    x = data[inputFea]
    y1 = data[outputFea1]  # 前轴承温度
    y2 = data[outputFea2]  # 后轴承温度

    # 训练神经网络多层感知器 - 回归模型，测试模型，预测前轴承温度
    model_mlp1 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3, 4, 5), activation='identity', max_iter=200,
                              random_state=1)
    model_mlp1.fit(x, y1)

    # 训练神经网络多层感知器 - 回归模型，测试模型，预测后轴承温度
    model_mlp2 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3, 4, 5), activation='identity', max_iter=200,
                              random_state=1)
    model_mlp2.fit(x, y2)

    # 归一化
    wtDf = pd.read_csv(wtPath, index_col=0)
    cols = df.columns.tolist()
    for idx, col in enumerate(cols):
        wtDf[col] = (wtDf[col] - mn[idx]) / std[idx]
    x = wtDf[inputFea]
    y1 = wtDf[outputFea1]  # 前轴承温度
    y2 = wtDf[outputFea2]  # 后轴承温度

    predict_y1 = model_mlp1.predict(x)
    predict_y2 = model_mlp2.predict(x)

    # 预测的前后轴承温差
    diff_predict = predict_y1 - predict_y2
    diff_ture = y1 - y2

    # 得到残差残差
    residuals = diff_predict - diff_ture

    # 保存残差输出结果的csv文件
    df = pd.DataFrame({'residuals': residuals})

    return df


def filter_df(df, medfiltLen):
    """
    根据指定的长度对一个dataframe所有列进行滤波

    Parameters
    -------
        df: dataframe.
        medfiltLen: int. 滤波长度
    return:
        df: dataframe. 平滑之后的数据
    """
    smlen = int((medfiltLen / 2) + 1)
    tmpdf = df.copy()
    columns = df.columns.tolist()
    df.loc[:, columns] = df.loc[:, columns].rolling(window=medfiltLen, center=True).median()
    df.iloc[0:smlen, :] = tmpdf.iloc[0:smlen, :]
    df.iloc[-smlen:, :] = tmpdf.iloc[-smlen:, :]
    return df


def ada_fdd(stationCode, inPath, outPath):
    """
    对一个指定的电站进行单机自适应故障检测

    Parameters
    -------
        stationCode: int. 电站的编号，例如，杨家湾电站编号为82
        inPath: string. 预处理之后的数据路径
        outPath: string.
    return:
        faultProb: dataframe. 以概率的形式返回检测的结果
    """

    logger = logging.getLogger(str(stationCode))
    logger.debug('start all_wind_turbine_residuals {}'.format(stationCode))
    logger = logging.getLogger(str(stationCode))

    #为每一台风机找其对应的邻居
    neighborsDict, WTs = cluster_wts_improved(inPath)

    #用于保存预测的残差
    allResiduals = pd.DataFrame()

    #对每台风机，预测前、后轴承温差的残差
    for wt in WTs:
        wtPath = inPath + '/' + wt + '.csv'
        neighbors = neighborsDict[wt]   #根据风机的设备名称获取其邻居
        df = merge_neighbors(wt, neighbors, inPath)  #合并该风机所有的邻居，以便对所有的残差统一进行异常判断
        residual = calculate_residuals(df, wt, wtPath)  #计算残差
        allResiduals[wt] = residual['residuals']  #合并残差

    # 根据残差的分布，计算故障严重程度的概率
    faultProb = faultDetection(allResiduals, WTs, neighborsDict, outPath)
    return faultProb


def faultDetection(allResiduals, WTs, neighborsDict, outPath):
    """
    根据残差，自动识别故障

    Parameters
    -------
        allResiduals: dataframe. 所有风机的残差数据
        WTs: list. 所有风机的列表
        neighborsDict: directory. key：一台风机的device code. value：这台风机的所有邻居风机
        outPath: string
    return:
        faultProb: dataframe. 得到故障的概率
    """

    #初始化设置
    medfiltLen = 72
    probEstimation = pd.DataFrame(index=allResiduals.index, columns=allResiduals.columns)
    probEstimation.index = pd.to_datetime(probEstimation.index)
    faultEstimation = pd.DataFrame(columns=allResiduals.columns)

    #对所有的风机，按天识别故障的概率
    for idx, wt in enumerate(WTs):
        df = allResiduals.loc[:, [wt]]

        #获取该风机所有的邻居对应的残擦
        neighbors = neighborsDict[wt]
        neighborsDf = allResiduals.loc[:, neighbors]
        mn = neighborsDf.stack().mean()
        sigma = neighborsDf.stack().std()

        #残差数据处理，移除无效数据，设置时间索引并排序，滤波
        df = df.dropna(axis=0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = filter_df(df, medfiltLen)

        #统计残差分布的均值和方差
        cdf = st.norm(mn, sigma)
        #计算故障严重程度的概率，并保存
        tmp = df.loc[df.index, wt].apply(lambda x: PHI2(x, cdf))
        probEstimation.loc[df.index, wt] = tmp
        tmpDf = probEstimation.loc[df.index, wt]
        tmpDf.fillna(method='ffill', inplace=True)
        tmpDf.fillna(method='bfill', inplace=True)
        tmpDf = tmpDf.groupby(df.index.to_period('D')).mean()
        faultEstimation[wt] = tmpDf

        #画图显示故障的严重程度
        plt.subplots(figsize=(18, 3))
        plt.plot(df.index, df)
        k = 3.5 #大于99.9%的概率展示
        plt.plot(df.index, [mn + k*sigma]*df.shape[0], 'r.')
        plt.plot(df.index, [mn - k*sigma] * df.shape[0], 'r.')
        plt.title(wt)
        plt.ylabel('Residuals')
        plt.xlabel('Time Stamp (10 min)')
        plt.savefig(outPath + '/'+wt + '_faultDetection.png', dpi=300)
        plt.show()

    faultEstimation.to_csv(outPath + '/adaFDDProbResults.csv')  # 保存每天的故障概率
    probEstimation.to_csv(outPath + '/probFDDEvery10Min.csv')  # 保存每10分钟的故障概率
    faultEstimation.index = faultEstimation.index.start_time
    return faultEstimation


def ann_single_equation(x, cof, intercepts):
    """
    求运行一次单层3个神经元神经网络ann模型得到的参数,其中公式为 y = h1*x1+h2*x2+h3*x3+b
    :param x: 输入变量
    :param cof:
    :param intercepts:
    :return: 包含权重和偏置的列表[h1,h2,h3,b]
    """
    # h用来存放最终公式中涉及的参数,w1为输入到隐含层的权重列,w2为隐含层到输出层的权重列
    #b1为输入到隐含层所需的偏置列,b2为隐含层到输出层的偏置列
    h = []
    w1 = cof[0]
    w2 = cof[1]
    b1 = intercepts[0]
    b2 = intercepts[1]

    #求h权重
    w22 = list()
    for i in range(len(w2)):
        w22.append(w2[i][0])
    for i in range(len(w1)):
        w = w1[i]
        h1 = w * w22
        hi = sum(h1)
        h.append(hi)

    #求b偏置
    b = list(b1 * w22)
    b.append(b2[0])
    b = sum(b)

    #包含权重和偏置的列表
    h.append(b)
    h_b = h

    return h_b


def PHI(x, cdf, mean):
    """
    根据累积概率密度函数，给出任意一个变量属于该分布的概率

    Parameters
    -------
        x: float.
        cdf: 累积概率密度函数
    return:
        a probablility: float. 变量属于该分布的概率
    """
    phi = cdf.cdf(x)
    if phi > 0.5:
        return phi
    else:
        return abs(1 - phi)


def PHI2(x, cdf):
    """
    根据累积概率密度函数，给出任意一个变量属于该分布的概率

    Parameters
    -------
        x: float.
        cdf: the cdf of a Gaussian
    return:
        a probablility: float. 变量属于该分布的概率
    """
    phi = cdf.cdf(x)
    return abs(2 * phi - 1)

def precision_recall(startDate, endDate):
    '''
    only test by zyy
    :param startDate:
    :param endDate:
    :return:
    '''


    siteDataDir = directory("{}/{}/{}".format('data', stationCode, 'siteData'))
    groundTruthPath = "{}/groundTruth.xlsx".format(siteDataDir)

    #get ground truth
    gd = pd.read_excel(groundTruthPath,index_col=0)
    gd = gd[(gd.record_time >= startDate) & (gd.record_time <= endDate)]
    gdDict = dict.fromkeys(gd.index.tolist())
    for wt in gdDict:
        gdDict[wt] = gd.at[wt,'record_time']

    #get detection results
    THR = 0.95
    adaProb = pd.read_csv(adaFDDResultsDataDir+'/adaFDDProbResults.csv',index_col = 0)
    adaProb = adaProb.iloc[1:-1]
    tmp = adaProb.stack()
    tmp = tmp[tmp>THR]
    detected = tmp.index.tolist()
    detectedNum = len(detected)
    dateCol = 0
    wtCol = 1
    TP = 0
    FN = len(gdDict)

    #get true positive number
    for idx,detecedSample in enumerate(detected):
        wt = detecedSample[wtCol]
        dt = pd.to_datetime(detecedSample[dateCol])
        if wt in gdDict.keys() and (gdDict[wt] - dt).days < 10 and (gdDict[wt] - dt).days >=0:
            TP = TP + 1
            FN = FN - 1
            print('TP',wt, dt,gdDict[wt],(gdDict[wt] - dt).days )

    precision = TP/detectedNum
    recall = TP/(TP+FN)


    print('precision',precision)
    print('recall',recall)

    return precision, recall
if __name__ == "__main__":
    #定义要检测和诊断的电站、故障的类型
    stationCode = 82 #桥头铺电站：83 #杨家湾电站：82
    fddType = 'generatorBearing'  #发电机轴承故障诊断
    startDate = '2018-09-23'  #诊断的起至时间
    endDate = '2018-12-30'

    #定义文件的路径
    adaFDDPreProcessedDataDir = directory(
        "{}/{}/{}/{}/{}".format('results', fddType, stationCode, 'preProcessedData', 'adaFDD'))  #预处理之后的数据路径
    cfFDDPreProcessedDataDir = directory(
        "{}/{}/{}/{}/{}".format('results', fddType, stationCode, 'preProcessedData', 'cfFDD'))
    resultsDir = directory("{}/{}/{}/{}".format('results', fddType, stationCode, 'results'))  #结果存放路径
    adaFDDResultsDataDir = directory("{}/{}/{}/{}/{}".format('results', fddType, stationCode, 'results', 'adaFDD'))
    cfFDDResultsDataDir = directory("{}/{}/{}/{}/{}".format('results', fddType, stationCode, 'results', 'cfFDD'))
    #测试该算法所需要的时间
    adaFDDPreProcessedDataDirTest = directory(
        "{}/{}/{}/{}/{}/{}".format('results', fddType, stationCode, 'preProcessedData', 'adaFDD','test'))

    #生成测试所需的起止时间内的数据
    flist = glob.glob(adaFDDPreProcessedDataDir + '/*.csv')
    for f in flist:
        fileName = os.path.basename(f)
        df = pd.read_csv(f,index_col=0)
        df.index = pd.DatetimeIndex(df.index)
        df = df[(df.index >startDate) & (df.index < endDate)]
        df.to_csv(adaFDDPreProcessedDataDirTest+'/'+fileName)

    #进行单机自适应故障诊断
    adaFDDProbResults = ada_fdd(stationCode, adaFDDPreProcessedDataDirTest, adaFDDResultsDataDir)




    #test for choosing the optimal period of data, using 10 days as an interval
    # interval = 10 #days
    # maxlen = 101 # the max number of period
    # precisionList =[]
    # recallList = []
    # for i in range(interval,maxlen,interval):
    #
    #     startDate = pd.to_datetime(endDate -datetime.timedelta(days=i))
    #     print(i,startDate,endDate)
    #
    #
    #     adaFDDProbResults = ada_fdd(stationCode, adaFDDPreProcessedDataDir, adaFDDResultsDataDir)
    #
    #     precision, recall = precision_recall(startDate, endDate)
    #     precisionList.append(precision)
    #     recallList.append(recall)
    #
    # print(precisionList,recallList)
    # plt.plot(precisionList,recallList,'r*')
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.xlabel('Precision')
    # plt.ylabel('Recall')
    # plt.savefig(adaFDDResultsDataDir + '/precision_recall.png', dpi=300)






