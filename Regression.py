# 回归问题
import torch
import torch.nn.functional as F  # 激励函数都在这
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


# 显示数据
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承 __init__ 功能
        super(Net, self).__init__()

        # 隐藏层线性输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)

        # Module 中的 forward 功能：

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数（隐藏层的线行值）
        x = self.predict(x)  # 输出值
        return x


# 定义网络结构：
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # 网络结构

# optimizer 是训练的工具：

# 传入 net 的所有参数, 学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# 预测值和真实值的误差计算公式 (均方差)
loss_func = torch.nn.MSELoss()

plt.ion()  # something about plotting

for t in range(200):
    # 喂给 net 训练数据 x, 输出预测值
    prediction = net(x)
    # 计算两者的误差: (1. nn output, 2. target)
    loss = loss_func(prediction, y)
    # 清空上一步的残余更新参数值
    optimizer.zero_grad()
    # 误差反向传播, 计算参数更新值
    loss.backward()
    # 将参数更新值施加到 net 的 parameters 上
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()