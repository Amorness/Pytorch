import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
import torchvision

BATCH_SIZE = 50

data_raw = pd.read_csv('./82/rawData/82M101M39M24.csv')
data_raw = data_raw.dropna()
data_raw = data_raw.reset_index(drop=True)
data_input = data_raw[['GN013','NC005','TR002']]

data_input = data_input.astype('float32')

data_StdInput = (data_input - data_input.min())/(data_input.max()-data_input.min())

data_output = data_raw['GN010']-data_raw['GN011']

data_output = data_output.astype('float32')
data_StdOutput = (data_output-data_output.min())/(data_output.max()-data_output.min())
data_StdOutput2 = data_StdOutput

data_StdInput = np.array(data_StdInput)
data_StdOutput = np.array(data_StdOutput)

data_StdInput = data_StdInput.reshape(-1,1,3)
data_StdOutput = data_StdOutput.reshape(-1,1,1)
data_StdInput = torch.from_numpy(data_StdInput)
data_StdOutput = torch.from_numpy(data_StdOutput)

torch_dataset = Data.TensorDataset(data_StdInput,data_StdOutput)

train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=3,
)


from torch import nn
from torch.autograd import Variable
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.out = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x , None)  # (seq, batch, hidden)
        x = self.out(x)
        return x
net = LSTM(3, 4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for e in range(10):
    for step, (b_x, b_y) in enumerate(train_loader):
        var_x = Variable(b_x)
        var_y = Variable(b_y)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (e + 1) % 100 == 0: # 每 100 次输出结果

    print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data.numpy()))

pred_test = net(data_StdInput) # 测试集的预测结果
# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# 画出实际结果和预测的结果
plt.figure(figsize=(50,5))
plt.title('82M101M39M24')
plt.plot(pred_test, 'r', label='prediction')
plt.plot(data_StdOutput2, 'b', label='real')
plt.legend(['prediction','real'],loc='best')
plt.show()
