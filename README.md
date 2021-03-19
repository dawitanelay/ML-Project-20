# University of Pisa Machine Learning Project 2020

### Abstract

Our goal was to build and compare different models from different software tools. We compare a
Neural Network from Keras [1] and both a Support Vector Machine and a K-nn regressors from the
Scikit-learn [2] framework. We perform model selection and validate via a combination of screening
phases and cross-validation, also testing on an internal test set extracted via hold-out.

### Introduction

The aims of project B of the Machine Learning (ML) course held by Professor Alessio Micheli at [Department of Computer Science](https://di.unipi.it/) of [University of Pisa](https://www.unipi.it/) is to bulid a ML model simulator to solve a multi-target regression task on CUP Dataset provided by the professor and Make a comparison among models .
The project contains two different datasets. The [Monk](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems) and Cup dataset. The Monk dataset is the bench mark dataset used to compare different models.

we  study the influence of the various hyper-parameters upon the implementation of a
neural network model. This is achieved through intensive use of screening phases which also provide
the added benefit of alleviating the computational burden of the many grid-searches.
The result of said effort, in the form of a neural network implementing a mini-batch Stochastic
Gradient Descent, is then put to the test through a comparison with two main other well known
regression models. The former, a Regressor-chain SVR, presents a Radial Basis Function kernel (rbf)
with a degree of polynomial kernel function and regularization parameters while the latter, a
K-Neighbors Regressor, weights the influence of each neighbor with respect to their euclidean
distance from the record fed to it.




- All the detalis about the project can be found on the full report [here](https://github.com/dawitanelay/ML-Project-20/blob/main/notebook/cupResult/smile_report.pdf)

### References
[1] https://keras.io/

[2] https://scikit-learn.org/stable/

[3] https://pandas.pydata.org/

[4] https://numpy.org/

[5] https://matplotlib.org/

[6] https://seaborn.pydata.org/

### Authors
- Dawit Mezemir Anelay  mailto:d.anelay1@studenti.unipi.it

