Instructions for Running
========================

Run the rogram as "python3 main.py <dataset> <classifier> <dimen_reduc>" 

<dataset> and <classifier> arguments are mandatory

<dimen_reduc> is optional. On specifying it, the denoted dimensionality reduction method
	will run. Otherwise, the classifier will run on the dataset with original dimensions

example 1: python3 main.py iris knnc pca
example 2: python3 main.py iris knnc

valid arguments
---------------
<dataset> => [iris, letter, pd_speech, kannada]
<classifier> => [knnc, svm, rf, xgboost, nn]
<dimen_reduc> => none or [sffs, mi, lsh, pca, rp]
