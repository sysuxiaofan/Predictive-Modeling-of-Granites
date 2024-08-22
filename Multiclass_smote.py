import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#Import raw data and extract trained features and labels
data= pd.read_csv('./multiclass_example_known.csv')
x=data.iloc[:,4:]
y=data.iloc[:,3]
y=pd.to_numeric( y, errors='coerce').fillna('0').astype('int32')
print(y)


from collections import Counter
from imblearn.over_sampling import  SMOTE

# 2.View samples of individual labels：
counter = Counter(np.array(y))
print(counter)

#3.Data set visualization：
plt.scatter(np.array(x)[:, -13], np.array(x)[:, 16], c=np.array(y),s=10)
plt.xlabel(x.columns[-13])
plt.ylabel(x.columns[16])
# plt.legend(loc="upper right",scatterpoints=3,fontsize=12)
plt.show()
plt.scatter(np.array(y), np.array(x)[:, 19],s=10)
plt.xlabel('type')
plt.ylabel(x.columns[19])
# plt.legend(loc="upper right",scatterpoints=3,fontsize=12)
plt.show()
#Converts dataframe data into array data
X=np.array(x)
Y=np.array(y)
# 4.2 SMOTE oversampling. We used the basic smote oversampling algorithm
X_resampled2, Y_resampled2 = SMOTE().fit_resample(X, Y)
counter_resampled2 = Counter(Y_resampled2)
print("SMOTE oversampling results：\n", counter_resampled2)
#
plt.scatter(X_resampled2[:, -13], X_resampled2[:, 16], c=Y_resampled2)
plt.xlabel('Cr')
plt.ylabel('Co')
plt.show()
 #Store the data after sampling
print(Y_resampled2[:])
sampling=np.column_stack((X_resampled2,Y_resampled2))
df=pd.DataFrame(sampling,columns=['B1','B4','B7','Ag','B','Al2O3','As','Au','Ba','Be','Bi','CaO','Cd','Co','Cr','Cu','F','Fe2O3','Hg','K2O','La',
                                  'Li','MgO','Mn','Mo','Na2O','Nb','Ni','P','Pb','Sb','SiO2','Sn','Sr','Th','Ti','U','V','W','Y2','Zn','Zr','Gravity','Magnetic',
                                  'type'])
df.to_csv( './oversampling_known3.csv',index=False)