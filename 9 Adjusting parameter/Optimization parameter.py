__Author__ = "Encounters Shen"

#自动寻找最优化参数的算法
# 1、网格搜索优化参数
# 2、随机搜索优化参数

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from pandas import read_csv
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

filename = "Pima Indians.csv"
dataset = read_csv(filename)

data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

# 网格搜索优化参数——通过遍历已定义参数的列表，来评估算法的参数，从而找到最优参数

model = Ridge()
param_grid = {'alpha':[1,0.1,0.01,0.001,0]}#可设定多个Key:Value对，同时查询多个参数的最优参数值
grid = GridSearchCV(estimator=model,param_grid=param_grid)
grid.fit(x,y)

print('最高得分：%.3f'%grid.best_score_)
print('最优参数：%s'%grid.best_estimator_.alpha)
print(grid.best_estimator_)
print(grid.best_params_)


# 随机搜索优化参数——采用随机采样分布的方式搜索合适的参数
# 与网格搜索优化参数相比，随机搜索优化参数提供了一种更高效的解决办法（特别是在参数数量多的情况下）
# 随机搜索优化参数为每个参数定义了一个分布函数，并在该空间中采样

'''
model = Ridge()
param_grid = {'alpha':uniform()}
grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=100,random_state=7)
grid.fit(x,y)
print('最高得分：%.3f'%grid.best_score_)
print('最优参数：%s'%grid.best_estimator_.alpha)
print(grid.best_estimator_)
print(grid.best_params_)
'''