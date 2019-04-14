__Author__ = "Encounters Shen"


# 分类算法的评估矩阵
# 1、分类准确度
# 2、对数损失函数（Logloss）
# 3、AUC图
# 4、混淆矩阵
# 5、分类报告（Classification Report）
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
filename = "Pima Indians.csv"
dataset = read_csv(filename)
data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

# 1、分类准确度
n_split = 10
seed = 7
kfold = KFold(n_splits=n_split,random_state=seed)
model1 = LogisticRegression()
result_scores1 = cross_val_score(model1,x,y,cv=kfold)
print("算法评估结果准确度：%.3f%%  %.3f%%"%(result_scores1.mean(),result_scores1.std()))

# 2、对数损失函数——对数损失函数越小，模型越好，而且使损失函数尽量是一个凸函数，便于收敛计算
model2 = LogisticRegression()
scoring = "neg_log_loss"
result_scores2 = cross_val_score(model2,x,y,cv=kfold,scoring=scoring)
print("算法评估结果准确度：%.3f%%  %.3f%%"%(result_scores2.mean(),result_scores2.std()))

# 3、AUC图
scoring1 = 'roc_auc'
model3 = LogisticRegression()
result_scores3 = cross_val_score(model3,x,y,cv=kfold,scoring=scoring1)
print("算法评估结果准确度：%.3f%%  %.3f%%"%(result_scores3.mean(),result_scores3.std()))

# 4、混淆矩阵
x_train,x_validation,y_train,y_validation = train_test_split(x,y,test_size=0.33,random_state=4)
model4 = LogisticRegression()
model4.fit(x_train,y_train)
prediction = model4.predict(x_validation)
matrix = confusion_matrix(y_validation,prediction)
classes = {0,1}
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)

# 5、分类报告
model5 = LogisticRegression()
model5.fit(x_train,y_train)
prediction1 = model5.predict(x_validation)
report = classification_report(y_validation,prediction)
print(report)