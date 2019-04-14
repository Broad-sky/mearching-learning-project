from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'Iris_Flower.csv'
names = ['separ_length','separ_width','petal_length','petal_width','class']
dataset = read_csv(filename,names=names)
# print(dataset) #数据集
# print(dataset.shape) #数据维度
print(dataset.head(6)) #显示前6条数据
# print(dataset.describe()) #统计描述数据

# count  150.000000  150.000000  150.000000  150.000000 行数
# mean     5.843333    3.054000    3.758667    1.198667 均值
# std      0.828066    0.433594    1.764420    0.763161 标准差
# min      4.300000    2.000000    1.000000    0.100000 最小值
# 25%      5.100000    2.800000    1.600000    0.300000 较低百分位数
# 50%      5.800000    3.000000    4.350000    1.300000 中位数
# 75%      6.400000    3.300000    5.100000    1.800000 较高百分位数
# max      7.900000    4.400000    6.900000    2.500000 最大值

print(dataset.groupby('class').size()) #数据分类分布

'''
# 箱线图
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()

#直方图
dataset.hist()
pyplot.show()

# 散点矩阵图
scatter_matrix(dataset)
pyplot.show()
'''

#分离评估数据集
array = dataset.values
# print(array)
X = array[:,0:4]
Y = array[:,4]
print(Y)
validation_size = 0.2
seed = 7
X_train,X_validation,Y_train,Y_validation \
    = train_test_split(X,Y,test_size=0.2,random_state=seed)
# print(X_train,Y_train)
print(X_train.shape,Y_train.shape,X_validation.shape,Y_validation.shape)

#创建模型
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

results = []
for m in models:
    kfold = KFold(n_splits=10,random_state=seed)
    cv_results = cross_val_score(models[m],X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)'%(m,cv_results.mean(),cv_results.std()))
print(results)

svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))