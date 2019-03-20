import numpy as np
import pandas as pd
data_raw = pd.read_csv('./example2.csv',index_col = u'Time')
data_cholesky=np.linalg.cholesky(data_raw.corr())
print(data_raw.corr())
print(data_cholesky)

import numpy as np
from time import time
S_0 = 100.0    # 股票或指数初始的价格;
K = 105        #  行权价格
T = 2.0        #  期权的到期年限(距离到期日时间间隔)
r = 0.05       #   无风险利率
sigma = 0.405   # 波动率(收益标准差)
M = 50         # number of time steps 时间周期分成50份
dt = T/M       # time interval 时间间隔
I = 20000       # number of simulation
S = np.zeros((M+1, I))
S[0] = S_0
np.random.seed(2000)
start = time()
for t in range(1, M+1):
    z = np.random.standard_normal(I)
    S[t] = S[t-1] * np.exp((r- 0.5 * sigma **2)* dt + sigma * np.sqrt(dt)*z)
C_0 = np.exp(-r * T)* np.sum(np.maximum(S[-1] - K, 0))/I
end = time()
print ('total time is %.6f seconds'%(end-start))
print ('European Option Value %.6f'%C_0)
# 前２０条模拟路径
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.grid(True)
plt.xlabel('Time step')
plt.ylabel('index level')
print
for i in range(100):
    plt.plot(S.T[i])
plt.show()

print(len(S.T[1]))
