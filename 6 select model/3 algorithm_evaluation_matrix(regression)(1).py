__Author__ = "Encounters Shen"

# 回归算法矩阵
# 1、平均绝对误差
# 2、均方误差
# 3、决定系数（R平方）
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
filename = "Hoston House Price.csv"
dataset = pd.read_csv(filename)
print(dataset.head(10))
data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

kfold = KFold(n_splits=10,random_state=7)
model = LinearRegression()
# 1、平均绝对误差
scoring = 'neg_mean_absolute_error'
result_scores = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
print("MAE：%.3f  %.3f" % (result_scores.mean(),result_scores.std()))

# 2、均方误差
scoring1 = 'neg_mean_squared_error'
result_scores1 = cross_val_score(model,x,y,cv=kfold,scoring=scoring1)
print("MSE:%.3f  %.3f"%(result_scores1.mean(),result_scores1.std()))

# 3、决定系数（R平方）
scoring2 = 'r2'
result_scores2 = cross_val_score(model,x,y,cv=kfold,scoring=scoring2)
print("R2: %.3f  %.3f"%(result_scores2.mean(),result_scores2.std()))