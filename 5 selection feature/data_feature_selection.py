__Author__ = "Encounters Shen"
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  #卡方检验
import matplotlib.pyplot as plt
from numpy import set_printoptions
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

filename = "Pima Indians.csv"
dataset = read_csv(filename)

data = dataset.values
x = data[:,0:-1]
y = data[:,-1]
# lis = data.columns.tolist()
# print(x)
# print(x.shape[1])

#1、单变量特征选择
#对于分类问题(y离散),可采用卡方检验，f_classif,mutual_info_classif,互信息
#对于回归问题(y连续),可采用皮尔逊相关系数，f_regression,mutual_infp_regression,最大信息系数
selector = SelectKBest(score_func=chi2,k=4)
print("selector=",selector)
fit = selector.fit(x,y)
# print(fit.pvalues_)
score = fit.scores_
print(score)
# new_x = fit.transform(x)
# print(new_x)
new_x = selector.fit_transform(x,y)
print(new_x)

plt.bar(range(x.shape[1]),score)
plt.show()

# fit()  ---->transform()   和   fit_transform()    是一样的效果

# 2、递归特征消除——递归特征消除(RFE)使用一个基模型来进行多轮训练，每轮训练后消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
# 通过每一个基模型的精度，找到对最终的预测结果影响最大的数据特征。
model = LogisticRegression()
rfe = RFE(model,4)
fit = rfe.fit(x,y)
print("特征个数：")
print(fit.n_features_)
print("被选定的特征：")
print(fit.support_)
print("特征排名：")
print(fit.ranking_)

# 3、主要成分分析——是使用线性代数来转换压缩数据，通常被称作数据降维。除了PCA，还有LDA（线性判别分析）
pca = PCA(n_components=4)
fit = pca.fit(x)
print("解释方差：%s" %fit.explained_variance_ratio_)
print(fit.components_)

# 4、特征重要性——袋装决策树算法（Bagged Decision Tress）、随机森林算法、极端随机树算法都可以用来计算数据特征的重要性。
# 这三个算法都是集成算法中的袋装算法。
model = ExtraTreesClassifier()
fit = model.fit(x,y)
print("特征重要度：",fit.feature_importances_)