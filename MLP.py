import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchvision
from sklearn.neural_network import MLPRegressor


for i in range(1,25):
    data_raw = pd.read_csv('./82/rawData/82M101M39M'+ str(i) +'.csv', index_col=0)
    data_raw = data_raw.dropna()
    data_raw = data_raw.reset_index(drop=True)
    inputFea = data_raw[['TR002', 'NC005', 'GN013']]  # 分别对应有功功率、机舱温度、u相温度
    inputFea = (inputFea - inputFea.min()) / (inputFea.max() - inputFea.min())

    outputFea1 = data_raw['GN010']  # 发电机前轴承温度

    outputFea2 = data_raw['GN011']  # 发电机后轴承温度

    diff_ture = outputFea1 - outputFea2
    diff_ture = (diff_ture - diff_ture.min()) / (diff_ture.max() - diff_ture.min())

    # 归一化
    # data = data_raw.apply(lambda x: (x - np.mean(x)) / np.std(x))
    # mn = data_raw.mean().tolist()
    # std = data_raw.std().tolist()
    # 如果没有u相温度，则移除之
    # if 'GN013' not in data.columns.tolist():
    #     inputFea.remove('GN013')
    #     print('This site DONOT collect generators U temperature. That may effect the performance of FDD model.')

    # 定义输入、输出
    x = inputFea
    y = diff_ture
    # y1 = outputFea1  # 前轴承温度
    # y2 = outputFea2  # 后轴承温度

    # 训练神经网络多层感知器 - 回归模型，测试模型，预测前轴承温度
    model_mlp1 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3, 4, 5), activation='identity', max_iter=200,
                              random_state=1)
    model_mlp1.fit(x, y)

    # # 训练神经网络多层感知器 - 回归模型，测试模型，预测后轴承温度
    # model_mlp2 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3, 4, 5), activation='identity', max_iter=200,
    #                           random_state=1)
    # model_mlp2.fit(x, y2)

    # # 归一化
    # wtDf = pd.read_csv('./82/rawData/82M101M39M1.csv', index_col=0)
    # cols = data_raw.columns.tolist()
    # for idx, col in enumerate(cols):
    #     wtDf[col] = (wtDf[col] - mn[idx]) / std[idx]
    # x = inputFea
    # y1 = outputFea1  # 前轴承温度
    # y2 = outputFea2  # 后轴承温度
    #
    # predict_y1 = model_mlp1.predict(x)
    # predict_y2 = model_mlp2.predict(x)
    #
    # # 预测的前后轴承温差
    # diff_predict = predict_y1 - predict_y2
    # diff_ture = y1 - y2
    # residuals = diff_predict - diff_ture

    predict_y = model_mlp1.predict(x)

    residuals = predict_y - diff_ture

    print(residuals)

    plt.figure(figsize=(50, 5))
    plt.title('82M101M39M'+ str(i) +'-MLP')
    plt.ylim([-0.5, 1])
    plt.plot(predict_y, 'r', label='prediction')
    plt.plot(diff_ture, 'b', label='real')
    plt.plot(residuals, 'g', label='residuals')

    plt.legend(['prediction', 'real','residuals'], loc='best')
    plt.savefig('./MLP Figure/82M101M39M'+ str(i) +'-MLP.png')
    plt.show()