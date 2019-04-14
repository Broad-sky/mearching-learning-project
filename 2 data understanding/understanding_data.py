__Author__ = "Encounters Shen"

from pandas import read_csv
from pandas import set_option
set_option('precision',2)

filename = "creditcard.csv"
# names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
#        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
#        'Class']
dataset = read_csv(filename)

#数据理解
# 1、简单查看数据
print(dataset.head(10))
# 2、数据的维度
print(dataset.shape)
# 3、数据的类型
print(dataset.dtypes)
# 4、描述性统计
print(dataset.describe())
# 5、数据分组分布
print(dataset.groupby('Class').size())
# 6、数据属性的相关性
print(dataset.corr(method='pearson'))
# 7、数据的分布分析（计算所有数据属性的高斯分布偏离情况）
print(dataset.skew())

