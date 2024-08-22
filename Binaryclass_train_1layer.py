import os
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,roc_auc_score,auc,precision_recall_curve
import keras
from keras import layers
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.inspection import permutation_importance
# Import normalized data
data=pd.read_csv('./train_normal.csv')
data1=pd.read_csv('./test_normal.csv')
print(data)
#Partition the eigenmatrices data_X,data_Y
X_train=data.iloc[:,:-1]
print(X_train)
Y_train=data.iloc[:,-1]
print(Y_train)
X_test=data1.iloc[:,:-1]
print(X_train)
Y_test=data1.iloc[:,-1]
print(Y_train)
np.random.seed(3)
scoring = {'roc_auc':'roc_auc','accuracy':'accuracy', 'precision':'precision','recall':'recall','f1':'f1'}
def aupr(y_true,y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    roc_aupr = auc(recall,precision)
    return roc_aupr
def result_print(fit, x_train, y_train, x_test, y_test):
    from sklearn.preprocessing import OrdinalEncoder
    model=fit.best_estimator_.model
    y_train_pred = model.predict_classes(x_train)  # 用best_estimator预测y_train_pred
    y_train_pred1=model.predict(x_train)
 #   y_train_pred_ba = np.argmax(np.array(y_train_pred), axis=1)
    print(y_train_pred,model.predict(x_train))

    y_test_pred = model.predict_classes(x_test)  # 用best_estimator预测y_test_pred
    y_test_pred1 = model.predict(x_test)  # 用best_estimator预测y_test_pred1
 #   y_test_pred_ba = np.argmax(np.array(y_test_pred), axis=1)
    print(y_test_pred,y_test)
    cv_results = pd.DataFrame(fit.cv_results_).set_index(['params'])
#    print(cv_results)
    cv_results_mean = cv_results[
        ['mean_train_accuracy', 'mean_train_f1', 'mean_train_precision', 'mean_train_recall', 'mean_train_roc_auc',
         'mean_test_accuracy', 'mean_test_f1', 'mean_test_precision', 'mean_test_recall', 'mean_test_roc_auc']]
    cv_results_std = cv_results[
        ['std_train_accuracy', 'std_train_f1', 'std_train_precision', 'std_train_recall', 'std_train_roc_auc',
         'std_test_accuracy', 'std_test_f1', 'std_test_precision', 'std_test_recall', 'std_test_roc_auc']]

    print('Best cv_roc_auc: %f using %s' % (fit.best_score_, fit.best_params_))
    print(cv_results_mean)
    # print(cv_results_std)

    train_score_list = []
    test_score_list = []
    score_list = []
    model_metrics_name = [accuracy_score, precision_score, recall_score, f1_score]
    for matrix in model_metrics_name:
        train_score = matrix(y_train, y_train_pred)
        test_score = matrix(y_test, y_test_pred)
        train_score_list.append(train_score)
        test_score_list.append(test_score)
    model_metrics_name2=[aupr,roc_auc_score]
    for matrix in model_metrics_name2:
        train_score = matrix(y_train, y_train_pred1)
        test_score = matrix(y_test, y_test_pred1)
        train_score_list.append(train_score)
        test_score_list.append(test_score)
    score_list.append(train_score_list)
    score_list.append(test_score_list)
    score_df = pd.DataFrame(score_list, index=['train', 'test'],
                            columns=['accuracy', 'precision', 'recall', 'f1', 'aupr', 'roc_auc'])
    print("Best: %f using %s" % (fit.best_score_, fit.best_params_))
    print('test_METRICS:')
    print(score_df)
    measure_result = classification_report(y_test,y_test_pred,output_dict=True)
    print('measure_result = \n', measure_result)
    print("-------------------------------------------------------------------")
    # cv_results_mean.to_csv('./结果1层/acc.csv')
    # score_df.to_csv('./结果1层/hunxiao.csv')
    # roc曲线
    # fpr, tpr, threshold = roc_curve(y_test, y_test_pred1)
    # roc_auc = auc(fpr, tpr)  # 准确率代表所有正确的占所有数据的比值
    # print('roc_auc:', roc_auc)
    # lw = 2
    # plt.subplot(1, 1, 1)
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('1 - specificity')
    # plt.ylabel('Sensitivity')
    # plt.title('ROC', y=0.5)
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # result = permutation_importance(fit, x_test, y_test, n_repeats=10, random_state=42)
    # sorted_idx = result.importances_mean.argsort()
    #
    # plt.barh(range(X_test.shape[1]), result.importances_mean[sorted_idx])
    # plt.yticks(range(X_test.shape[1]), [X_test.columns[i] for i in sorted_idx])
    # plt.xlabel('Permutation Importance')
    # plt.show()

#Grid search
import numpy
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
def build_model(n_neurons1=30,n_neurons2=30,input_shape=[44]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(keras.layers.Dense(n_neurons1, activation="relu",kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(1,activation="sigmoid"))
    optimizer = keras.optimizers.adam()
    model.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=['acc'])
    return model

keras_reg = KerasClassifier(build_model,verbose=0)
param_distribs = {
    "n_neurons1": [140,160],
    "epochs": [50]
}


grid_search = GridSearchCV(keras_reg, param_distribs, cv=5,scoring=scoring,
                       refit='roc_auc',return_train_score=True)
grid_result=grid_search.fit(X_train,Y_train)

result_print(grid_result,X_train,Y_train,X_test,Y_test)


#Prediction of unknown lithology

unkown=pd.read_csv('./unknown_normal.csv')
unkown_X=unkown.iloc[:,:-1]
predictions1 = model.predict(unkown_X)
y_1= np.argmax(predictions1, axis=1)
print(predictions1)
y_last = pd.Series(y_1)
print(y_last)
unkown['type_predict']=y_last
print(unkown)
unkown.to_csv('./predict.csv',index=False,header=True)
