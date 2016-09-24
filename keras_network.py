import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Dense, Merge, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping

from keras_classifier import *
from data_generator import get_training_data

train = get_training_data(mergeMutated=True)

X = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
Y = le.transform(train['species'])

"Standardized the data with unit variance and unit mean"
"Use this scalar to standardize the test data also"
scaler = StandardScaler().fit(X)


seed = 1337
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.10, random_state=seed)


print("Load Test Data...")
test = pd.read_csv("data/test.csv")
test_id = test['id']
X_test = test.drop(['id'], axis=1).values


X_train = scaler.transform(X_train)
try:
    X_valid= scaler.transform(X_valid)
except ValueError:
    pass
X_test = scaler.transform(X_test)


print(X_train.shape)
print(X_test.shape)
input_dim = np.int32(X_train.shape[1] / 3)

X_margin_train = X_train[:, input_dim*0:input_dim*1]
X_shape_train = X_train[:, input_dim:input_dim*2]
X_texture_train = X_train[:, input_dim*2:input_dim*3]

X_margin_valid = X_valid[:, input_dim*0:input_dim*1]
X_shape_valid = X_valid[:, input_dim:input_dim*2]
X_texture_valid = X_valid[:, input_dim*2:input_dim*3]


X_margin_test = X_test[:, input_dim*0:input_dim*1]
X_shape_test = X_test[:, input_dim:input_dim*2]
X_texture_test = X_test[:, input_dim*2:input_dim*3]

def create_model(level_1_neuron_count, level_2_neuron_count, level_3_neuron_count, dropout_prob, activation, init_weights):

    print("Creating the model...")
    print("Neuron Level 1 Count: %d" % level_1_neuron_count)
    print("Neuron Level 2 Count: %d" % level_2_neuron_count)
    print("Neuron Level 3 Count: %d" % level_3_neuron_count)
    print("Dropout Probability: %f" % dropout_prob)
    print("Activation Function: %s" % activation)
    print("Weight Initialization Scheme: %s" % init_weights)

    margin_model = Sequential()
    margin_model.add(Dense(output_dim=level_1_neuron_count, input_dim=input_dim, init=init_weights))
    margin_model.add(Dropout(dropout_prob))

    shape_model = Sequential()
    shape_model.add(Dense(output_dim=level_1_neuron_count, input_dim=input_dim, init=init_weights))
    shape_model.add(Dropout(dropout_prob))

    texture_model = Sequential()
    texture_model.add(Dense(output_dim=level_1_neuron_count, input_dim=input_dim, init=init_weights))
    texture_model.add(Dropout(dropout_prob))

    model = Sequential()
    model.add(Merge([margin_model, shape_model, texture_model], mode='concat', concat_axis=1))

    if level_2_neuron_count != 0:
        model.add(Activation(activation))
        model.add(Dropout(dropout_prob))
        model.add(Dense(level_2_neuron_count, init=init_weights))

    if level_3_neuron_count != 0:
        model.add(Activation(activation))
        model.add(Dropout(dropout_prob))
        model.add(Dense(level_3_neuron_count, init=init_weights))

    model.add(Activation(activation))
    model.add(Dense(99, init=init_weights))
    model.add(Activation('softmax'))
    print("Compiling model")
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def margin_shape_texture_split(X):
    assert(X.shape[1] == np.int32(input_dim*3))
    return np.hsplit(X, 3)



def grid_search_cv():
    model = KerasClassifier(build_fn=create_model, X_transform_fn=margin_shape_texture_split, nb_epoch=75, batch_size=128, verbose=1)

    old_param_grid = [
        {
            "level_1_neuron_count" : [192, 256, 384],
            "level_2_neuron_count" : [384, 512, 768],
            "level_3_neuron_count": [196],
            "dropout_prob" : np.linspace(0.2, 0.4, num=3),
            "init_weights" :  ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal','he_uniform']
        }
    ]

    param_grid = [
        {
            "level_1_neuron_count": [384],
            "level_2_neuron_count": [384],
            "level_3_neuron_count": [0, 128, 196, 256],
            "dropout_prob": [0.3],
            "activation" : ["tanh"],
            "init_weights" :  ['he_uniform']
        }
    ]

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
    grid_result = grid.fit(X_train, Y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

def run_kaggle():
    model = create_model(level_1_neuron_count=384*2.5, level_2_neuron_count=384*2.5, level_3_neuron_count=196*2.5, dropout_prob=0.3, activation="tanh", init_weights="he_uniform")

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    print("Fit \n")
    model.fit([X_margin_train, X_shape_train, X_texture_train],
              Y_train,
              validation_data=([X_margin_valid, X_shape_valid, X_texture_valid], Y_valid),
              callbacks=[earlyStopping],
              nb_epoch=500, batch_size=64, verbose=1)

    print("Prediction \n")
    Y_test = model.predict_proba([X_margin_test, X_shape_test, X_texture_test])

    print(Y_test[1])

    submission = pd.DataFrame(Y_test, index=test_id, columns=le.classes_)
    submission.to_csv('data/submission_NN.csv')

if __name__ == '__main__':
    run_kaggle()
    # grid_search_cv()


