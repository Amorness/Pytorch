import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate

# show data
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, (h_n,c_n) = self.rnn(x ,None)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None  # for initial hidden state

plt.figure(1, figsize=(30, 5))
plt.ion()  # continuously plot

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction = rnn(x)  # rnn output
    # !! next step is important !!
    #h_state = h_state.data  # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)  # calculate loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)

plt.ioff()
plt.show()