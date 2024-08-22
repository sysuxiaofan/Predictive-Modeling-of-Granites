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

def build_model(n_neurons1=30,input_shape=[44]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(n_neurons1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(3,activation="softmax"))
    optimizer = keras.optimizers.adam()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['acc'])
    return model
keras_reg = KerasClassifier(build_model,verbose=0)
param_distribs = {
        "n_neurons1": [10,20,30,40,50,60,70,80,90]}
grid_search = GridSearchCV(keras_reg, param_distribs, cv=5)
grid_result=grid_search.fit(X_train,Y_train)
print("Optimal parameters of the model：",grid_search.best_params_)
print("Optimal model score：",grid_search.best_score_)
print("Optimal model object：",grid_search.best_estimator_)


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
n_neurons1=[10,20,30,40,50,60,70,80,90]
plt.plot(n_neurons1,means)
plt.show()
# c=pd.merge(pd.DataFrame(params),pd.DataFrame({'acc':means,}),left_index=True,right_index=True)
# c.to_csv('C:/Users/WKQ/Desktop/tangao代码/结果1层/acc.csv')
#Feature importance ranking
model = build_model(n_neurons1=50,input_shape=[44])
X_train2,X_vaild,Y_train2,Y_vaild  = train_test_split(X_train,Y_train,test_size=0.2, random_state=1)
Hit=model.fit(X_train2,Y_train2,validation_data=(X_vaild,Y_vaild),epochs=10,verbose=0)
val_acc=Hit.history["val_acc"][-1]
print(val_acc)
pd.DataFrame(Hit.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
results = []
print(' Computing MPL feature importance...')
for k in range(len(COLS)):
#
    save_col = X_train2
    x=save_col.drop([COLS[k]],axis=1)
    save_col2=X_vaild
    x2=save_col2.drop([COLS[k]],axis=1)
    #print(x)
    #print(X_test)
    # oof_preds = model.predict(X_test, verbose=0).squeeze()
    # mae = np.sum(np.mean(np.abs(oof_preds - Y_test)))
    model2=build_model(n_neurons1=50,input_shape=[43])
    Hit1=model2.fit(x,Y_train2,validation_data=(x2,Y_vaild),epochs=10,verbose=0)
    val_acc1=Hit1.history["val_acc"][-1]
    difference=val_acc-val_acc1
    results.append({'feature': COLS[k], 'mae': difference})
df = pd.DataFrame(results)
df = df.sort_values('mae')
plt.figure(figsize=(20,10))
plt.bar(np.arange(len(COLS)),df.mae)
plt.xticks(np.arange(len(COLS)),df.feature.values)
plt.title('MPL Feature Importance',size=16)
plt.xlim((-1,len(COLS)))
plt.show()
# #
#Predict test sets with models
test_predictisons=model.predict(X_test)
print(test_predictisons)
# Go back to sequential encoding
from sklearn.preprocessing import OrdinalEncoder
Y_predict=np.argmax(np.array(test_predictisons),axis=1)
Y_test2=np.argmax(np.array(Y_test),axis=1)
# print(Y_predict)

# Generate confusion matrix and model evaluation score
measure_result = classification_report(Y_test2, Y_predict)
print('measure_result = \n', measure_result)
print("-------------------------------------------------------------------")
#roc curve
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# lw=2
# for i in range(3):
#     fpr[i], tpr[i], _ = roc_curve(Y_test.values[:, i], test_predictisons[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(3), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
# print(roc_auc)
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()
#Prediction of unknown lithology

unkown=pd.read_csv('./unknown_nomal.csv')
unkown_X=unkown.iloc[:,:-1]
predictions1 = model.predict(unkown_X)
y_1= np.argmax(predictions1, axis=1)
print(predictions1)
y_last = pd.Series(y_1)
print(y_last)
y_last.to_csv('./Multiclass_predict.csv',mode='a+',index=False,header='type_predict')
