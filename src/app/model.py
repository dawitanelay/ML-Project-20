
#importing libraries
import numpy as np
import pandas as pd
#keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from keras.wrappers.scikit_learn import KerasClassifier




from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from keras import regularizers
from keras.layers import Dropout

# for combining all the hyper-parameters
import itertools 
from time import time



# Grid search for the best parameter
g_eta = [0.1,0.2,0.3]
g_batchSize = [10,20,25,30]
g_hiddeLayerunit1 = [4,8,10]
g_momentum = [0.6,0.8,0.9]
g_afHiddenLayerunit1 = ["relu", "tanh"]


# reading training and testing data from the csv file
trainDf = pd.read_csv ('data/csv/monks-3Train.csv',header=None).to_numpy()
testDf = pd.read_csv ('data/csv/monks-3Test.csv',header=None).to_numpy()
trainDf = minmax_scale(trainDf, feature_range=(0,1), axis=0)
testDf = minmax_scale(trainDf, feature_range=(0,1), axis=0)

#Preparing trainDf for training
trainX = trainDf[:, 1:7]
trainY = trainDf[:, 0]
#trainX = scaler.fit_transform(trainX)
#trainY = scaler.fit_transform(trainY)

#Preparing testDf for validation
testX = testDf[:, 1:7]
testY = testDf[:, 0]

def create_model(lr=0.1,momentum=0.9,g_hiddeLayerunit1=4,g_hiddeLayerunit2=4,activation1="tanh",activation2="tanh",activation3="sigmoid",init_mode='uniform', g_decay=0.01):
    model = Sequential()
    model.add(Dense(g_hiddeLayerunit1, input_dim=6, kernel_initializer=init_mode, activation=activation1))
    model.add(Dropout(0.4)),
    model.add(Dense(g_hiddeLayerunit2, kernel_initializer=init_mode, activation=activation2,kernel_regularizer=regularizers.l1(0.01))) # l1 regularization parameter could also be something we can also grid search for.
    model.add(Dropout(0.4)),
    model.add(Dense(1, activation3))
    sgd = SGD(lr=lr, momentum=momentum, decay=g_decay, nesterov=False,)  # We can add decay to hyper parameter list to get optimum value. 
    model.compile(optimizer=sgd, loss='mean_squared_error',metrics=['accuracy'])
    return model

start=time()
model = KerasClassifier(build_fn=create_model)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
g_hiddeLayerunit1=[4,8,16]
g_hiddeLayerunit2=[4,8,16]
g_decay = [0.01] # more estimations can be added
activation1 = ["relu", "tanh","sigmoid","softmax"]
activation2 = ["relu", "tanh","sigmoid","softmax"]
activation3 = ["relu", "tanh","sigmoid","softmax"]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
lr=np.arange(0.1, 0.9, 0.1).tolist()
momentum=np.arange(0.1, 0.9, 0.1).tolist()
param_grid = dict(
    lr=lr,
    g_hiddeLayerunit1=g_hiddeLayerunit1,
    g_hiddeLayerunit2=g_hiddeLayerunit2,
    activation1=activation1,
    activation2=activation2,
    activation3=activation3,
    batch_size=batch_size,
    init_mode=init_mode,
    g_decay = g_decay,
    epochs=epochs)


grid = GridSearchCV(estimator=model, param_grid=param_grid,n_jobs=-1,cv=3,verbose=2)



grid_result = grid.fit(trainX, trainY)

end=time()

print("Total Running Time: %f",end-start)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Mean\tSTD\tParams")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) 