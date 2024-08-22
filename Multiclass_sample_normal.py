import numpy as np
import pandas as pd
from collections import Counter
#Import samples that are oversampled and to be predicted respectively
oversample=pd.read_csv('./oversampling_known3.csv')
sample=pd.read_csv('./multiclass_example_unknown.csv')
print(oversample)
#Convert the tag to one-hot encoding
oversample['type']=pd.factorize(oversample["type"])[0].astype(int)
oversample = oversample.join(pd.get_dummies(oversample.type))
del oversample["type"]
#Divide the eigenmatrix and label data_X,data_Yt
X_data=oversample.iloc[:,:-3]
print(X_data)
Y_data=oversample.iloc[:,-3:]
print(Y_data)
np.random.seed(3)
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test  = train_test_split(X_data,Y_data,test_size=0.2, random_state=10,stratify=Y_data)
counter0 = Counter(np.array(Y_test[0]))
counter1 = Counter(np.array(Y_test[1]))
counter2 = Counter(np.array(Y_test[2]))
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
#Normalization of unknown samples
x2=sample.iloc[:,2:-1]
x2 -= mean
x2 /= std
x2=x2.join(sample.type)
print(x2)
# Write the normalized oversampled data to the csv file, train.csv,test.csv,unknown_nomal.csv
X_train.to_csv('./train.csv',index=False)
X_test.to_csv('./test.csv',index=False)
x2.to_csv('./unknown_nomal.csv',index=False)
# np.savetxt( "oversample1_normal.csv", np.array(x1), delimiter="," )
