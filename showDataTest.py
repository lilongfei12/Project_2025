import pandas as pd

#利用pandas读取数据
df = pd.read_csv('data/diabetes.csv')

#打印头部数据可以知道数据的基本形式和特征个数
print(df)