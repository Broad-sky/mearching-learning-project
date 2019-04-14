__Author__ = "Encounters Shen"
# 有一些标准的流程可以实现对机器学习问题的自动化处理，在scikit-learn中通过Pineline来定义和自动化运行这些流程
# 在机器学习方面有一些可以采用的标准化流程，这些标准化流程是从共同的问题中提炼出来的，例如评估框架中的数据缺失等
# Pineline能够从数据转换导评估模型的整个机器学习流程进行自动化处理
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest

filename = "Pima Indians.csv"
dataset = read_csv(filename)
print(dataset.head())

data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

# 1、数据准备和生成模型的Pineline
# 两部：（1）正太化数据 （2）训练一个线性判别分析模型
'''
num_fold = 10
seed = 7
kfold = KFold(n_splits=num_fold,random_state=seed)
steps = []
steps.append(("Standardize",StandardScaler()))
steps.append(("lda",LinearDiscriminantAnalysis()))
print("steps:\n",steps)
model = Pipeline(steps)
print("model:\n",model)
results = cross_val_score(model,x,y,cv=kfold)
print(results.mean())
'''

# 2、特征选择和生成模型的Pipeline
# （1）通过主成分分析进行特征选择 （2）通过统计进行特征选择（3）特征集合 （4）生成一个逻辑回归模型
num_fold = 10
seed = 7
kfold = KFold(n_splits=num_fold,random_state=seed)

# 生成FeatureUnion
features = []
features.append(("pca",PCA()))
features.append(("select_best",SelectKBest(k=6)))
print("features:\n",features)
# 生成Pioeline
steps = []
steps.append(("feature_union",FeatureUnion(features)))
steps.append(("logistic",LogisticRegression()))
print("steps:\n",steps)
model = Pipeline(steps)
result = cross_val_score(model,x,y,cv=kfold)
print(result.mean())