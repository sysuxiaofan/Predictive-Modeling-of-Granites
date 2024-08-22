import numpy as np
import pandas as pd
from collections import Counter
# Import known and unknown samples
oversample=pd.read_csv('./binary_example_known.csv')
sample=pd.read_csv('./binary_example_unknown.csv')
print(sample)

# Partition the eigenmatrices data_X,data_Y
X_data=oversample.iloc[:,:-1]
print(X_data)
Y_data=oversample.iloc[:,-1:]
print(Y_data)
np.random.seed(3)
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test  = train_test_split(X_data,Y_data,test_size=0.2, random_state=10)
Y_train=1-Y_train
Y_test=1-Y_test
counter0 = Counter(np.array(Y_train['type']))
counter1 = Counter(np.array(Y_test['type']))
print(counter0)
print(counter1)

#The data features of the training set are normalized
mean = np.mean(X_train, axis=0)  # 均值
std = np.std(X_train, axis=0) #方差/标准差
X_train-= mean  # 训练集
X_train /= std  # 训练集
X_train=X_train.join(Y_train)

print(X_train)
X_test -= mean
X_test /= std
X_test=X_test.join(Y_test)
print(X_test)
# Do normalization for unknown samples
x2=sample.iloc[:,2:-1]
print(x2,mean)
x2 -= mean
x2 /= std
print(x2)
x2=x2.join(sample.iloc[:,-1])
print(x2)
# Write normalized oversampled data to csv file;
X_train.to_csv('./train_normal.csv',index=False)
X_test.to_csv('./test_normal.csv',index=False)
x2.to_csv('./unknown_normal.csv',index=False)
