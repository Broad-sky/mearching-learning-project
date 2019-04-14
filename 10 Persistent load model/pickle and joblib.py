__Author__ = "Encounters Shen"
# 在实际的目中，需要将生成的模型序列化，并将其发布到生产环境
# 当有新数据出现时，需要反序列化已保存的模型，然后用其预测新的数据

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

file_name = "Pima Indians.csv"
dataset = read_csv(file_name)
data = dataset.values
x = data[:,0:-1]
y = data[:,-1]

test_size = 0.33
seed = 4
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=seed)
model = LogisticRegression()
model.fit(x_train,y_train)

'''
# pickle是标准的Python序列化方法，可以通过它来序列化机器学习算法生成的模型，并将其保存到文件中
# 保存模型
model_file = 'finalized_model.save'
with open(model_file,'wb') as model_f:
    dump(model,model_f)

# 加载模型
with open(model_file,'rb') as model_f:
    loaded_model = load(model_f)
    print(loaded_model)
    result = loaded_model.score(x_test,y_test)
    print("算法评估结果：%.3f%%" % (result*100))
'''

# joblib是Scipy生态环境的一部分，提供了通用的工具来序列化Python的对象和反序列化Python的对象
model_file = 'finalized_model.savjoblib'
with open(model_file,'wb') as model_f:
    dump(model,model_f)

with open(model_file,'rb') as model_f:
    loaded_model = load(model_f)
    result = loaded_model.score(x_test,y_test)
    print("算法评估结果：%.3f%%"%(result*100))

# 在生成机器学习模型时，需要考虑以下几个问题
# 1、Python的版本：要记录Python的版本，大部分情况下
# 在序列化模型的和反序列化模型时，需要使用相同的Python版本
# 2、类库版本：同样需要记录所有的主要类库的版本
# 因为序列化模型和反序列化模型时需要使用相同版本的类库，
# 不仅需要Scipy和scikit-learn
# 版本一致，也其他类库版本也需要一致
# 3、手动序列化：有时需要手动序列化算法参数，这样可直接在scikit-learn中或其他的
# 平台重现这个模型。我们通常会花费大量的时间在选择算法和参数调整上
# 将这个过程手动记录下来比序列化模型更有价值
