__Author__ = "Encounters Shen"

# 评估算法的方法：1、分离训练集和评估数据集；2、K折交叉验证分离；3、弃一交叉验证分离；4、重复随机评估、训练数据集分离

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
filename = "Pima Indians.csv"
dataset = read_csv(filename)
data = dataset.values
x = data[:,0:-1].shape
x1 = data[:,:].shape
print("x: ",x)
print("y: ",x1)
y = data[:,-1]
print(y)
print(dataset.dtypes)
print(dataset.head(10))
print(data)
#1、分离训练数据集和评估数据集
'''
test_size = 0.33
seed = 4#数据随机的粒度
x_train,x_validation,y_train,y_validation = train_test_split(x,y,test_size=test_size,random_state=seed)
print(x_train.shape)
print(x_validation.shape)
model = LogisticRegression()
model.fit(x_train,y_train)
result_scores = model.score(x_validation,y_validation)
print("算法评估结果: %.3f%%"%(result_scores*100))
'''
#2、k折交叉验证分离
# 通常会取K=3，5，10  一般情况下取10
'''
test_size = 0.33
seed = 4#数据随机的粒度
x_train,x_validation,y_train,y_validation = train_test_split(x,y,test_size=test_size,random_state=seed)
kfold = KFold(n_splits=10)
model = LogisticRegression()
result_scores = cross_val_score(model,x_train,y_train,cv=kfold)
print(result_scores)
print("算法评估结果：%.3f%%  %.3f%%"%(result_scores.mean()*100,result_scores.std()*100))
'''

#3、弃一交叉验证
'''
# 优点：（1）每一回合几乎所有的样本皆用于训练模型，因此最接近原始样本的分布，这样评估所得的结果比较可靠
#      （2）实验过程中没有随机因素会影响实验数据，确保实验过程是可以被复制的
# 缺点：（1）计算成本高，模型的数量和样本的数量相当。适合小样本
loocv = LeaveOneOut()
model = LogisticRegression()
result_scores = cross_val_score(model,x,y,cv=loocv)
print(result_scores)
print("算法评估结果：%.3f%%  %.3f%%"%(result_scores.mean()*100,result_scores.std()*100))
'''
#4、重复随机分离评估数据集与训练数据集
'''
kfold = ShuffleSplit(n_splits=10,test_size=0.33,random_state=7)
model = LogisticRegression()
result_scores = cross_val_score(model,x,y,cv=kfold)
print("算法评估结果：%.3f%%  %.3f%%"%(result_scores.mean()*100,result_scores.std()*100))
'''
