__Author__ = "Encounters Shen"

# 特征选择是困难耗时的，也需要对需求的理解和专业知识的掌握。在机器学习的应用开发中，最基础的是特征工程。    --吴恩达

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

filename = "Pima Indians.csv"
dataset = read_csv(filename)
# print(dataset.head(10))

#准备数据
arr=dataset.values
x = arr[:,0:8]
y = arr[:,-1]
# print(x)
# print(y)
#用fit()函数来准备数据转换的参数，然后调用transform()函数来做数据的预处理
#1、Rescale Data 调整数据尺度
print("type_x:",type(x))
print("x:",x)
transformer = MinMaxScaler(feature_range=(0,1))
new_x = transformer.fit_transform(x)
set_printoptions(precision=3)
print("new_x",new_x)

#2、Standardize Data 正太化数据
transformer1 = StandardScaler().fit(x)
new_x1 = transformer1.transform(x)
print("new_x1",new_x1)

#3、Normalize Data 标准化数据——归一元数据，适合处理稀疏数据（具有很多为0数据）
#  归一元处理的数据对使用权重输入的神经网路和使用K近邻算法的准确度有显著的提升
transformer2 = Normalizer().fit(x)
new_x2 = transformer2.transform(x)
print("new_x2",new_x2)

#4、Binarize Data 二值数据——是使用值将数据二值化，大于阈值设置为1，小于阈值设置为0
transformer3 = Binarizer().fit(x)
new_x3 = transformer3.transform(x)
print("new_x3",new_x3)
