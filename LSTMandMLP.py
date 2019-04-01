"""
Created on 2019-04-01 12:21:58

@author: yujie

     通过每台风机的机舱内温度、有功功率、U相定子温度，来预测发电机前后轴承温度差

     预测方法：（1）LSTM   （2）MLPRegressor


     输入：杨家湾电站每台风机的机舱内温度、有功功率、U相定子温度
     输出：发电机前后轴承温度差

     结果：利用LSTM和MLPRegressor两种情况下发电机前后轴承温度差预测值的残差进行比对


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torch import nn
from torch.autograd import Variable
from sklearn.neural_network import MLPRegressor

# Hyper Parameters
BATCH_SIZE = 50 # 批训练的数据个数
INPUTFILEPATH = './82/rawData' #数据读取路径
OUTPUTFILEPATH = './Residuals Figure' #输出路径
LR = 0.001  #学习率

def LSTM(i,data_input,data_ture):
    """
    通过LSTM对 提供的相关属性的真实值 与 目标属性的真实值 进行计算，得到目标属性的预测值

    :param i: 风机编号
    :param data_input : 需要参与LSTM的属性
    :param data_ture : 需要LSTM预测的属性的真实值
    :return: 发电机前后轴承温度差的实际值与预测值的残差
    """

    data_output = data_ture

    #转换为torch需要的类型
    data_input = data_input.astype('float32')
    data_output = data_output.astype('float32')

    #最大最小值归一化
    data_StdInput = (data_input - data_input.min()) / (data_input.max() - data_input.min())
    data_StdOutput = (data_output - data_output.min()) / (data_output.max() - data_output.min())

    #存储float32类型 且 归一化之后的目标属性真实值
    diff_ture = data_StdOutput


    data_StdInput = np.array(data_StdInput)
    data_StdOutput = np.array(data_StdOutput)

    #转换成torch需要的三维结构
    data_StdInput = data_StdInput.reshape(-1, 1, 3)
    data_StdOutput = data_StdOutput.reshape(-1, 1, 1)

    #转换成Pytorch的张量Tensors形式
    data_StdInput = torch.from_numpy(data_StdInput)
    data_StdOutput = torch.from_numpy(data_StdOutput)

    # 先转换成 Pytorch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(data_StdInput, data_StdOutput)
    # 把 dataset 放入 DataLoader
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,   #每一步训练的最小个数
        shuffle=True,            # 打乱数据
        num_workers=3,           # 多线程来读数据
    )

    class LSTM(nn.Module): # 继承 torch 的 Module
        #这里预先定义了output_size为1 也就是我们只需要预测一个值 ；num_layers表示有几层 RNN layers
        def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
            super(LSTM, self).__init__()

            # 定义每层用什么样的形式
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # 隐藏层使用LSTM
            self.out = nn.Linear(hidden_size, output_size)  # 输出层线性输出

        # 正向传播输入值, 神经网络分析出输出值
        def forward(self, x):
            # x shape (batch, time_step, input_size)
            # r_out shape (batch, time_step, output_size)
            # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
            # h_c shape (n_layers, batch, hidden_size)   同h_n
            x, (h_n, h_c) = self.rnn(x, None)  # None 表示初始 hidden state 会用全0的状态表示
            x = self.out(x)    # 输出值
            return x

    net = LSTM(3, 4) #定义神经网络 输入属性个数3   神经元个数4
    criterion = nn.MSELoss() #定义损失函数 均方差函数
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)#定义优化器 使用adam方法优化 学习率为0.001

    for e in range(10):     # 训练所有数据 10 次
        for step, (b_x, b_y) in enumerate(train_loader):  # 每一步 loader 释放一小批数据用来学习
            #转换成variable类，其中该类包含了三个属性 data(数据)、gtad(梯度)、grad_fn(funciton对象，用于反向传播)
            var_x = Variable(b_x)
            var_y = Variable(b_y)
            # 前向传播
            out = net(var_x)
            loss = criterion(out, var_y) #计算预测值与真实值之间均方差
            # 反向传播
            optimizer.zero_grad() #清空下一次传播使用的梯度
            loss.backward()  #反向传播 计算梯度
            optimizer.step()  #应用梯度

        #每一次训练控制台输出一次提醒
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data.numpy()))

    #读取张量Tensor类型的预测结果
    pred_test = net(data_StdInput)

    # 改变输出的格式
    pred_test = pred_test.view(-1).data.numpy()

    #计算预测值与真实值之间的残差
    residuals = pred_test - diff_ture

    return residuals


def MLP(i,data_input,data_ture):

    inputFea = data_input
    diff_ture = data_ture

    # 最大最小归一化
    inputFea = (inputFea - inputFea.min()) / (inputFea.max() - inputFea.min())
    diff_ture = (diff_ture - diff_ture.min()) / (diff_ture.max() - diff_ture.min())

    # 定义输入、输出
    x = inputFea
    y = diff_ture

    # 训练神经网络多层感知器 - 回归模型，测试模型，预测前轴承温度
    model_mlp1 = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3, 4, 5), activation='identity', max_iter=200, random_state=1)
    model_mlp1.fit(x, y)

    #预测前后轴承温差
    predict_y = model_mlp1.predict(x)

    #计算预测值与真实值之间的残差
    residuals = predict_y - diff_ture

    return residuals



if __name__ == "__main__":
    inputpath = INPUTFILEPATH
    outputpath = OUTPUTFILEPATH
    #读取24台风机数据
    for i in range(1,25):
        data = pd.read_csv(inputpath + '/82M101M39M' + str(i) + '.csv', index_col=0)
        #去除空值
        data = data.dropna()
        data = data.reset_index(drop=True)
        #选择输入属性：u相温度、机舱温度、有功功率
        data_input = data[['GN013', 'NC005', 'TR002']]

        #目标属性：发电机前后轴承温度差
        data_output = data['GN010'] - data['GN011']

        #计算lstm和mlp的残差
        residuals_lstm = LSTM(i, data_input,data_output)
        residuals_mlp = MLP(i, data_input,data_output)

        #计算残差绝对值
        residuals_lstmabs = residuals_lstm.abs()
        residuals_mlpabs = residuals_mlp.abs()

        print('lstm残差绝对值均值：',residuals_lstmabs.mean())
        print('mlp残差绝对值均值',residuals_mlpabs.mean())
        print('lstm残差均值:', residuals_lstm.mean())
        print('mlp残差均值:', residuals_mlp.mean())

        #作图功能，不做介绍 ，爱咋画咋画
        plt.figure(figsize=(50, 5))
        plt.title('82M101M39M' + str(i) + '-Residuals')
        plt.ylim([-0.5, 0.5])
        plt.plot(residuals_lstm, 'r', label='lstm')
        plt.plot(residuals_mlp, 'b', label='mlp')
        plt.legend(['lstm', 'mlp'], loc='best')
        plt.text(50000, 0.3, "lstm:" + str('%.10f' % residuals_lstm.mean()))
        plt.text(50000, 0.25, "mlp :" + str('%.10f' % residuals_mlp.mean()))
        plt.text(50000, 0.2, "lstmabs:" + str('%.10f' % residuals_lstmabs.mean()))
        plt.text(50000, 0.15, "mlpabs :" + str('%.10f' % residuals_mlpabs.mean()))
        plt.savefig(OUTPUTFILEPATH + '/82M101M39M' + str(i) + '-Residuals.png')
        plt.show()




