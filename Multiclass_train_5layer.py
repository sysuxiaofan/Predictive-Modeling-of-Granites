import keras
from keras import layers
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
warnings.filterwarnings("ignore")

# Import normalized data
data=pd.read_csv('./train.csv')
data1=pd.read_csv('./test.csv')
print(data)
X_train=data.iloc[:,:-3]
print(X_train)
Y_train=data.iloc[:,-3:]
print(Y_train)
X_test=data1.iloc[:,:-3]
print(X_train)
Y_test=data1.iloc[:,-3:]
print(Y_train)
#Grid search
def build_model(n_neurons1=30,n_neurons2=30, n_neurons3=10,n_neurons4=10,n_neurons5=10,input_shape=[44]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(n_neurons1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(n_neurons2, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(n_neurons3, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(n_neurons4, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(n_neurons5, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(3,activation="softmax"))
    optimizer = keras.optimizers.adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['acc'])
    return model

keras_reg = KerasClassifier(build_model,epochs=10,verbose=0)
param_distribs = {
    "n_neurons1": [10,20,30,40,50,60,70,80,90],
    "n_neurons2": [30],
    "n_neurons3": [10,20,30,40,50,60,70,80,90],
    "n_neurons4": [10,20,30,40,50,60,70,80,90],
    "n_neurons5": [10,20,30,40,50,60,70,80,90]}

a={"n_neurons1":[50],"n_neurons2":[80,90],"n_neurons3":param_distribs["n_neurons3"],"n_neurons4":param_distribs["n_neurons4"],
   "n_neurons5":param_distribs["n_neurons5"]}
grid_search = GridSearchCV(keras_reg, param_distribs, cv=5)

grid_result=grid_search.fit(X_train,Y_train)
# 返回参数组合
# 网格搜索训练后的副产品
print("Optimal parameters of the model：",grid_search.best_params_)
print("Optimal model score：",grid_search.best_score_)
print("Optimal model object：",grid_search.best_estimator_)


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
c=pd.merge(pd.DataFrame(params),pd.DataFrame({'acc':means,}),left_index=True,right_index=True)
c.to_csv('./结果5层/acc.csv',mode='a',header=False)
#Predict test sets with models
# test_predictisons=model.predict(X_test)
# print(test_predictisons)
# Go back to sequential encoding
# from sklearn.preprocessing import OrdinalEncoder
# Y_predict=np.argmax(np.array(test_predictisons),axis=1)
# Y_test2=np.argmax(np.array(Y_test),axis=1)
# # print(Y_predict)
#
# Generate confusion matrix and model evaluation score
# measure_result = classification_report(Y_test2, Y_predict)
# print('measure_result = \n', measure_result)
# print("-------------------------------------------------------------------")
#roc curve
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# lw=2
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(Y_test.values[:, i], test_predictisons[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# print("两层auc值:",roc_auc)
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(3), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()

