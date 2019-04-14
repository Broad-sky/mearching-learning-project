__Author__ = "Encounters Shen"

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from  sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


filename = "Pima Indians.csv"
dataset = read_csv(filename)

data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=seed)

#装袋（Bagging）算法
# 1、
# 决策树（装袋）
# cart = DecisionTreeClassifier()
# num_tree = 100
# model = BaggingClassifier(cart,n_estimators=num_tree)

#2、随机森林
# model = RandomForestClassifier(n_estimators=100,random_state=7,max_features=3)

#3、极端随机树
# model = ExtraTreesClassifier(n_estimators=100,random_state=7,max_features=7)

#提升算法
# model = AdaBoostClassifier(n_estimators=100,random_state=7)

#随机梯度提升
# model = GradientBoostingClassifier(n_estimators=100,random_state=7)

# 投票算法
models = []
model_logistic = LogisticRegression()
models.append(("logistic",model_logistic))
model_cart = DecisionTreeClassifier()
models.append(("cart",model_cart))
model_svc = SVC()
models.append(("svm",model_svc))
ensemble_model = VotingClassifier(estimators=models)
print(ensemble_model)
score = cross_val_score(ensemble_model,x,y,cv=kfold)
print(score.mean())