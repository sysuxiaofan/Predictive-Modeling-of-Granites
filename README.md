# Predictive-Modeling-of-Granites
These codes are used for data-driven expeditious mapping and identifying granites in covered area via deep machine learning
Usage

1)	Build the basic python runtime file and install the Dependencies in the Dependencies.txt file. For testing purposes choose binary_example_known.csv,binary_example_unknown,multiclass_example_known.csv,multiclass_example_unknown.csv from the  repository and put them in the same path as the code.

2)	In pycharm and other programming software, first run binary_normal.py to divide and normalize the training set and test set, and at the same time normalize the prediction sample. The normalized training set train_normal.csv, test set test_normal.csv and prediction set unknown_normal.csv were obtained.

3)	Then run Binaryclass_train_1layer.py, Binaryclass_train_2layer.py, Binaryclass_train_3layer.py, Binaryclass_train_4layer.py, Bi naryclass_train_5layer.py, so as to train one, two, three, four, five layers of neural networks, and obtain the grid search results and the corresponding prediction results of each layer of neural networks. The output file is predict.csv.

4)	Run Multiclass_smote.py to oversample known samples of multiple classes and save the output as oversampling_known3.csv.

5)	Run multiclass_sample_normal.py to divide the oversampling results from step 4 into a training set and a test set, and normalize the training set, test set, and prediction set. The output is train.csv, test.csv, unknown_nomal.csv.

6)	Then run Multiclass_train_1layer.py, Multiclass_train_2layer.py, Multiclass_train_3layer.py, Multiclass_train_4layer.py, Multic lass_train_5layer.py, so as to train one, two, three, four, five layers of neural networks, and obtain the grid search results and the corresponding prediction results of each layer of neural networks. The output file is Multiclass_predict.csv.

Quick test file

binary_example_known.csv: A sample of known labels in a binary classification, including a training set and a test set

binary_example_unknown.csv:Sample of unknown label in binary classification

multiclass_example_known.csv: Samples of known labels in multiple categories, including training sets and test sets

multiclass_example_unknown.csv:Sample of unknown labels in multiple categories
