__Author__ = "Encounters Shen"

import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np

filename = "Hoston House Price.csv"
dataset = read_csv(filename)
# dataset['TIME'] = pd.to_datetime(dataset['TIME'],unit='s')
# dataset['TIME'] = dataset['TIME'].dt.hour
# dataset['TIME'] = dataset['TIME'].dt.year
# dataset['TIME'] = dataset['TIME'].dt.month

print(dataset.head(10))
print(dataset.tail(10))
print(dataset.columns)
print(len(dataset.columns))
'''
# 绘制直方图--质量分布图，横轴表示数据类型，纵轴表示分布情况（可直观的看见数据是高斯分布，还是指数分布，还是偏态分布）
dataset.hist()
plt.show()

# 绘制密度图--是一种表现与数据值对应的边界或域对象的图形表示方法，一般用于呈现连续变量
dataset.plot(kind ='density',subplots = True,sharex= False)
plt.show()

# 绘制箱线图--盒须图，是一种非常好的用于显示数据分布状况的手段。中位数，上四分位数，下四分位数，上边缘，下边缘，边缘之外的异常值
dataset.plot(kind = 'box',subplots = True,sharex = False)
plt.show()
'''

# 相关矩阵图
correlations = dataset.corr(method='pearson')
print(correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(len(dataset.columns))
ax.set_xticks(ticks)#设置x轴或者y轴只显示哪些刻度
ax.set_yticks(ticks)
ax.set_xticklabels(dataset.columns,rotation=90)#设置刻度标签,rotation,fontsize
ax.set_yticklabels(dataset.columns)
plt.show()

#散点矩阵图
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
# plt.show()