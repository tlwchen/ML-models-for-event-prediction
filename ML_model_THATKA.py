#!/usr/bin/env python
# coding: utf-8

# # Models
# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import datetime, os
import sklearn

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold,train_test_split,GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, brier_score_loss, accuracy_score, make_scorer
from sklearn.calibration import calibration_curve
import sympy
import pyreadstat


from scipy.stats import linregress

import matplotlib.pyplot as plt
import pickle
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers


import keras_tuner as kt

from sklearn.model_selection import KFold, StratifiedKFold


from scipy.interpolate import interp1d




# ## Functions for Models
# ### ANN

# In[3]:


#set up model
def ANN_model(hp):
    # change to number of columms 
    #sets up layers
    
    # hp is probably keras_tuner.HyperParameters()
    # without specification it is just the default load, I guess
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(units = x_train.shape[1], activation='relu', input_dim=x_train.shape[1]))
    
    #0 to 2 number of hidden layers
    hp_layers = hp.Int("num_layers", 0, 2)
    if hp_layers != 0:
        for i in range(hp_layers):
            model.add(tf.keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i+1}", min_value=8, max_value=24, step=4),
                    activation=hp.Choice("activation", ["relu", "tanh"])
                )
            )

    # hp_units = hp.Int('units', min_value=0, max_value=32, step=8)  #adjust the max min and steps for hidden layer neuron numbers 
    # if hp_units != 0: 
    #     model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    
    model.add(tf.keras.layers.Dropout(hp.Float("dropout_", 0, 0.3, step=0.05))) #adjust dropout - randomly setting a fraction rate of inputs to 0 to prevent overfitting
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC'])
    return model



#tune hidden layer number, hidden layer neurons, activation, dropout, learning rate


#hyperband iteration is similar concept to k-fold 
#modifiy max_epochs, hyperband iteration according to data structure
def ANN_tuner(ANN_model, x_train, y_train):
    tuner = kt.Hyperband(ANN_model,
                         objective=kt.Objective("val_auc", direction="max"),
                         max_epochs= 30,
                         factor=5,
                         hyperband_iterations=3,
                         directory='my_dir',
                         project_name='intro_to_kt',overwrite=True)
    
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience= 3, mode = 'max')
    
    #add  [tensorboard_callback] to callbacks to see progress of tunings, need to load tensorboard when running
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose = 1)

    results = tuner.results_summary(num_trials = 1)
    print(results)
    
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    

#     print(f"""
#     The hyperparameter search is complete. The optimal number of units in the first densely-connected
#     layer is {best_hps.get('units')}, optimum drop is {best_hps.get('dropout_')}, and the optimal learning rate for the optimizer
#     is {best_hps.get('learning_rate')}.
#     """)
    
    return tuner, best_hps

#tune epoch number
def epoch_tuner(model, x_train, y_train):
    #finding optimum epochs to train model

    history = model.fit(x=x_train, 
            y=y_train, 
            epochs=50, 
            validation_split = 0.2, verbose = 0)

    val_auc_per_epoch = history.history['val_auc']
    best_epoch = val_auc_per_epoch.index(max(val_auc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    return best_epoch

#train 
def ANN_train(ANN_model,x_train, y_train, x_test):
    #finding best hyperparameter and epoch
    tuner, best_hps = ANN_tuner(ANN_model, x_train, y_train)
    model = tuner.hypermodel.build(best_hps)
    
    best_epoch = epoch_tuner(model, x_train, y_train)
    
    #Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    
    ## Retrain the model - not using kfold cv
    #  hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2, verbose = 0)

    #5-fold cross validation by manually splitting training set

    kfold = StratifiedKFold(n_splits=5, shuffle=True)    
    df = pd.DataFrame(columns = ['acc', 'auc', 'slope','intercept','Brier'], index=['0','1','2','3','4'])
    count = 0

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        hypermodel.fit(x_train5, y_train5, epochs=best_epoch)
        scores = hypermodel.evaluate(x_test5, y_test5, verbose=0)
        
        coor_y, coor_x=calibration_curve(y_test5, hypermodel.predict(x_test5), n_bins=20)
        df.iloc[count, 2], df.iloc[count,3],r_value,p_value,std_err=linregress(coor_x,coor_y)
        df.iloc[count, 4]=brier_score_loss(y_test5, hypermodel.predict(x_test5))
        
        print("%s: %.2f%%" % (hypermodel.metrics_names[1], scores[1]*100))
        print("%s: %.2f%%" % (hypermodel.metrics_names[2], scores[2]*100))
        
        df.iloc[count, 0]=scores[1] * 100
        df.iloc[count, 1]=scores[2] * 100
        count = count + 1
        
    
    y_score = hypermodel.predict(x_test)
    filename = y_train.name.replace("", "") + '_ANN'
    hypermodel.save(filename)
    filename = filename + '.pkl'
    filename_iv = y_train.name.replace("", "") + '_ANN_iv.pkl'
    
    f=open(filename_iv,"wb")
    pickle.dump(df,f)
    f.close()
    
#   y_predict = y_score
#   y_predict[y_predict>0.5]=1
#   y_predict[y_predict<=0.5]=0
    
  # print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename,




    

# --------------------------Artificial Neural Network
# ----------------------------------------------------------------------------
def ANN(x_tune,y_tune,x_train,y_train,x_test,search):
    from sklearn.neural_network import MLPClassifier

    #'hidden_layer_sizes': [x for x in product(range(1,100), range(1,3))]

    mlp = MLPClassifier(max_iter=5000, random_state=1)

    # Coarse search via RandomizedSearchCV (three layers for binary classification),
    # Number of the neurons range between 1 to 2 times of the feature number
    layer_n_cs=5
    initial_neuron=50
    list_a=[]
    for i in range(2,layer_n_cs,1):
        temp=[]
        for j in range(1,i,1):
            temp.append(initial_neuron)
        list_a.append(temp)

    parameter_space = {
        'hidden_layer_sizes': list_a,
        'activation': ['tanh','relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001,0.001,0.01,0.1], #regularization term combats overfitting
        'learning_rate': ['constant','invscaling','adaptive']}


    clf = RandomizedSearchCV(mlp, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5)
    clf.fit(x_tune, y_tune)

    #print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    Score=clf.best_score_


    # After knowing the number of layers
    layer_fix=2
    min_neuron=45
    max_neuron=85
    interv=5
    import auto_annln as annln
    list_ln=annln.fixlayer_adjneuron(layer_fix,min_neuron,max_neuron,interv)

    parameter_space = {
        'hidden_layer_sizes': list_ln,
        'activation': ['tanh'],
        'solver': ['sgd'],
        'alpha': [0.1],
        'learning_rate': ['constant']}
    clf = RandomizedSearchCV(mlp, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5)
    clf.fit(x_tune, y_tune)

    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    Score=clf.best_score_


    # Fine search via GridSearchCV
    parameter_space = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (50,100,100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [1e-200,1e-150,1e-100],
        'learning_rate': ['constant','invscaling','adaptive']}
    clf = GridSearchCV(mlp, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5)
    clf.fit(x_tune, y_tune)

    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    Score=clf.best_score_


    # formal________________________________________________________________________
    ann_model=MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'],
                            activation=best_params['activation'],
                            solver=best_params['solver'],
                            alpha=best_params['alpha'],
                            learning_rate=best_params['learning_rate'], max_iter=5000, random_state=1)

    #ann_model.fit(x_train, y_train)
    #y_predict=ann_model.predict(x_test)


    cv_solver=cross_validate(ann_model, x_train, y_train, cv=5, scoring="roc_auc", 
                             return_estimator=True)

    solver_list=cv_solver['estimator']
    ann_model=solver_list[-1]
    
    
    filename = y_tune.name.replace(" ", "") + '_ANN.pkl'
    joblib.dump(ann_model, filename)
    #ann_model=pickle.load(open('30readmin_annmodel','rb'))
    #solver_list=pickle.load(open('30readmin_annmodel_raw','rb'))
    
    y_score=ann_model.predict_proba(x_test)[:,1]
    y_predict=ann_model.predict(x_test)
    return y_predict, y_score

# ### Random Forest

# In[4]:


def RF(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    rf=RandomForestClassifier(bootstrap=True, oob_score=True, random_state=1)
    parameter_space = {
        'n_estimators':np.arange(5,100,10), 'min_samples_leaf':range(2,50,2)}

    if search == "grid":
        clf = GridSearchCV(rf, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    else:
        clf = RandomizedSearchCV(rf, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    clf.fit(x_tune, y_tune)
    
    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    Score=clf.best_score_


    rf_model=RandomForestClassifier(n_estimators=best_params['n_estimators'], min_samples_leaf=best_params['min_samples_leaf'], 
                max_features=10, bootstrap=False)
    
#    cv_solver=cross_validate(rf_model, x_train, y_train, cv=5, scoring="roc_auc", return_estimator=True)
#    solver_list=cv_solver['estimator']
#    rf_model=solver_list[-1]

     
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    df = pd.DataFrame(columns = ['acc', 'auc', 'slope','intercept','Brier'], index=['0','1','2','3','4'])
    count = 0
    
    

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        rf_model.fit(x_train5, y_train5)
        
        coor_y, coor_x=calibration_curve(y_test5, rf_model.predict_proba(x_test5)[:,1], n_bins=20)
        df.iloc[count, 2], df.iloc[count,3],r_value,p_value,std_err=linregress(coor_x,coor_y)
        df.iloc[count, 4]=brier_score_loss(y_test5, rf_model.predict_proba(x_test5)[:,1])
        
        df.iloc[count, 0]=accuracy_score(y_test5, rf_model.predict(x_test5),normalize=True)
        df.iloc[count, 1]=roc_auc_score(y_test5, rf_model.predict_proba(x_test5)[:,1])
        count = count + 1

    # rf_model.fit(x_train,y_train)
    # For random forest,the model uses the bootstrap sample (subsample=2/3rd of 
    # the training data for regression and 70% for classification) to fit each decision tree
    # and validate the tree performance against to the excluded. Therefore, a CV similar
    # procedure is already built in a random forest model. Not needed to set up cv.
    # Also the reason why random forest in unlikely prone to overfit problems.

    # Validation: tests of the accuracy of the model predictions against the data that
    # it excludes from the training of the tree.

    filename = y_tune.name.replace(" ", "") + '_RF.pkl'
    joblib.dump(rf_model, filename)
    oob_score=rf_model.oob_score
    # oob_error=1-oob_score
    
    filename_iv = y_tune.name.replace(" ", "") + '_RF_iv.pkl'
    f=open(filename_iv,"wb")
    pickle.dump(df,f)
    f.close()

    # Important variable that will inform the next round of feature elimination
    feature_import=pd.DataFrame(rf_model.feature_importances_)
    feature_import=pd.concat([feature_import,pd.DataFrame(x_train.columns.values)],axis=1)
    feature_import.columns=['score','variable']
    feature_import.sort_values('score',ascending=False,inplace=True)
    
    filename_fi = y_tune.name.replace(" ", "") + '_RF_FeatureImportance.pkl'
    f=open(filename_fi,"wb")
    pickle.dump(feature_import,f)
    f.close()

    y_score=rf_model.predict_proba(x_test)[:,1]
    y_predict = rf_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename


# In[5]:


#Inspired by LightGBM - HistoGradientBoosting

def HGB(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn.ensemble import HistGradientBoostingClassifier
    ## Test this results=cross_val_score(GrandientBoostClassifier,x,y,cv=3): calculating the results
    hgb = HistGradientBoostingClassifier(random_state=1)
    # Read this: ubsamplefloat, default=1.0.The fraction of samples to be used for fitting the individual base learners. 
    # If smaller than 1.0 this results in Stochastic Gradient Boosting. 
    
    parameter_space = {'learning_rate': [0.001,0.01,0.1, 0.5],
        'max_iter':np.arange(80,140,20), 'max_leaf_nodes': [31, 36, 41, 46]}
    
# 'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40,180,1), 'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

    if search == "grid":
        clf = GridSearchCV(hgb, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    else:
        clf = RandomizedSearchCV(hgb, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    clf.fit(x_tune, y_tune)
    
    print('Best parameters found:\n', clf.best_params_, clf.best_score_)
    best_params=clf.best_params_
    best_score=clf.best_score_

    hgb_model=HistGradientBoostingClassifier(learning_rate=best_params['learning_rate'], 
                            max_iter=best_params['max_iter'],
                            max_leaf_nodes=best_params['max_leaf_nodes'],
                            random_state=1)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    df = pd.DataFrame(columns = ['acc', 'auc', 'slope','intercept','Brier'], index=['0','1','2','3','4'])
    count = 0
    

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        hgb_model.fit(x_train5, y_train5)
        
        coor_y, coor_x=calibration_curve(y_test5, hgb_model.predict_proba(x_test5)[:,1], n_bins=20)
        df.iloc[count, 2], df.iloc[count,3],r_value,p_value,std_err=linregress(coor_x,coor_y)
        df.iloc[count, 4]=brier_score_loss(y_test5, hgb_model.predict_proba(x_test5)[:,1])
        
        df.iloc[count, 0]=accuracy_score(y_test5, hgb_model.predict(x_test5),normalize=True)
        df.iloc[count, 1]=roc_auc_score(y_test5, hgb_model.predict_proba(x_test5)[:,1])
        count = count + 1

    # cv_solver=cross_validate(hgb_model, x_train, y_train, cv=5, scoring="roc_auc", 
                             # return_estimator=True)
    
    # train_score=cv_solver['test_score']
    # solver_list=cv_solver['estimator']
    # hgb_model=solver_list[-1]
    
    filename = y_tune.name.replace(" ", "") + '_HGB.pkl'
    joblib.dump(hgb_model, filename)
    
    filename_iv = y_tune.name.replace(" ", "") + '_HGB_iv.pkl'
    f=open(filename_iv,"wb")
    pickle.dump(df,f)
    f.close()
    
    y_score=hgb_model.predict_proba(x_test)[:,1]
    y_predict = hgb_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename    


# In[6]:


def SGB(x_tune,y_tune,x_train,y_train,x_test,y_test,search):
    from sklearn.ensemble import GradientBoostingClassifier
    ## Test this results=cross_val_score(GrandientBoostClassifier,x,y,cv=3): calculating the results

    # Read this: ubsamplefloat, default=1.0.The fraction of samples to be used for fitting the individual base learners. 
    # If smaller than 1.0 this results in Stochastic Gradient Boosting. 
    sgb=GradientBoostingClassifier(random_state=1)
    parameter_space = {'learning_rate': [0.001,0.01,0.1],
        'n_estimators':np.arange(5,100,5), 'subsample': [0.6,0.7,0.8,0.9],
        'max_depth':range(2,15,1)}
    
    if search == "grid":
        clf = GridSearchCV(sgb, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    else:
        clf = RandomizedSearchCV(sgb, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    clf.fit(x_tune, y_tune)
    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    best_score=clf.best_score_

    sgb_model=GradientBoostingClassifier(learning_rate=best_params['learning_rate'], 
                            n_estimators=best_params['n_estimators'],
                            subsample=best_params['subsample'],
                            max_depth=best_params['max_depth'], random_state=1, 
                            max_features=10)

    cv_solver=cross_validate(sgb_model, x_train, y_train, cv=5, scoring="roc_auc", 
                             return_estimator=True)
    
    train_score=cv_solver['test_score']
    solver_list=cv_solver['estimator']
    sgb_model=solver_list[-1]
    
    filename = y_tune.name.replace(" ", "") + '_SGB.pkl'
    joblib.dump(sgb_model, os.path.join(output,filename))
    
    y_score=sgb_model.predict_proba(x_test)[:,1]
    
    y_predict = sgb_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename    


# ### Support Vector Machine

# In[2]:


def SVM(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn import svm
    from sklearn.calibration import CalibratedClassifierCV

    supportvm=svm.LinearSVC(dual=False, verbose=0)
    # The C parameter tells the SVM optimization how much you want to avoid misclassifying each training
    # example. For large values of C, the optimization will choose a smaller-margin hyperplane if that 
    # hyperplane does a better job of getting all the training points classified correctly.

    # The gamma parameter defines how far the influence of a single training example reaches, 
    # with low values meaning 'far' and high values meaning 'close'. 

    parameter_space={"penalty":['l1','l2'], "C":np.arange(0.1,0.8,0.02)}
    # "gamma": [0.01,0.05]
    if search == "grid":
        clf = GridSearchCV(supportvm, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    else:
        clf = RandomizedSearchCV(supportvm, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose=1)
    clf.fit(x_tune, y_tune)
    best_params=clf.best_params_
    

    # svm.SVC includes 5-fold cross-validation by default through enabling the function of probability 

    #svm_model=svm.SVC(C=120, gamma=5e-07, kernel='rbf', probability=True)
    svm_model=svm.LinearSVC(C=best_params['C'], penalty=best_params['penalty'], dual=False)
    cali=CalibratedClassifierCV(svm_model)
    #svm_model.fit(x_tune,y_tune)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cvacc = []
    cvauc = []

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        cali.fit(x_train5, y_train5)
        
        accuracy=accuracy_score(y_test5, cali.predict(x_test5),normalize=True)
        auc=roc_auc_score(y_test5, cali.predict_proba(x_test5)[:,1])
        
        cvacc.append(accuracy)
        cvauc.append(auc)
    dic={'acc':cvacc,'acu':cvauc}
    
    filename = y_tune.name.replace(" ", "") + '_SVM.pkl'
    joblib.dump(svm_model, os.path.join(output,filename))
    
    filename_iv = y_tune.name.replace(" ", "") + '_SVM_iv.pkl'
    f=open(filename_iv,"wb")
    pickle.dump(dic,f)
    f.close()

    y_score=svm_model.predict_proba(x_test)[:,1]
    y_predict = svm_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename

 

# ### K-Nearest Neighbour

# In[3]:


# --------------------------K-Nearest Neighbour
# ----------------------------------------------------------------------------
def KNN(x_tune,y_tune,x_train,y_train,x_test,y_test, search):
    from sklearn.neighbors import KNeighborsClassifier


    # KNN is the simplest model from the mathematical and programmatical aspects
    # Try implementing KNN++  
    knn=KNeighborsClassifier(algorithm='auto')
    parameter_space={"n_neighbors": [50,100,150,200,250,300,350], "weights": ["distance","uniform"],
                     "p": [1,2]}
    
    if search == "grid":
        clf = GridSearchCV(knn, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    else:
        clf = RandomizedSearchCV(knn, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    clf.fit(x_tune, y_tune)
    
    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    best_score=clf.best_score_

    knn_model=KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], 
                            weights=best_params['weights'],
                            p=best_params['p'],
                            algorithm='auto', n_jobs=-1)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    df = pd.DataFrame(columns = ['acc', 'auc', 'slope','intercept','Brier'], index=['0','1','2','3','4'])
    count = 0
    

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        knn_model.fit(x_train5, y_train5)
        
        coor_y, coor_x=calibration_curve(y_test5, knn_model.predict_proba(x_test5)[:,1], n_bins=20)
        df.iloc[count, 2], df.iloc[count,3],r_value,p_value,std_err=linregress(coor_x,coor_y)
        df.iloc[count, 4]=brier_score_loss(y_test5, knn_model.predict_proba(x_test5)[:,1])
        
        df.iloc[count, 0]=accuracy_score(y_test5, knn_model.predict(x_test5),normalize=True)
        df.iloc[count, 1]=roc_auc_score(y_test5, knn_model.predict_proba(x_test5)[:,1])
        count = count + 1


    # cv_solver=cross_validate(knn_model, x_train, y_train, cv=5, scoring="roc_auc", 
    #                          return_estimator=True)

    # train_score=cv_solver['test_score']
    # print("training scores are" + str(train_score))

    # solver_list=cv_solver['estimator']
    # knn_model=solver_list[-1]
    
    filename = y_tune.name.replace(" ", "") + '_KNN.pkl'
    joblib.dump(knn_model, filename)
    
    filename_iv = y_tune.name.replace(" ", "") + '_KNN_iv.pkl'
    f=open(filename_iv,"wb")
    pickle.dump(df,f)
    f.close()

    #y_predict=knn_model.predict(x_test)
    y_score=knn_model.predict_proba(x_test)[:,1]
    
    y_predict = knn_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename


# ### Net-Elastic Penalized LR

# In[4]:


def NEPLR(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn.linear_model import SGDClassifier
    nep=SGDClassifier(loss="log_loss", penalty="elasticnet", max_iter=5000)
    parameter_space={"alpha": [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2,2.5,3], "l1_ratio": [0.2,0.4,0.6,0.8]}
    
    if search == "grid":
        clf = GridSearchCV(nep, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    else:
        clf = RandomizedSearchCV(nep, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    clf.fit(x_tune, y_tune)
    

    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    best_score=clf.best_score_


    nep_model=SGDClassifier(alpha=best_params['alpha'], 
                           l1_ratio=best_params['l1_ratio'],
                           loss="log_loss", penalty="elasticnet", max_iter=5000)
    
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cvacc = []
    cvauc = []

    for train, test in kfold.split(x_train, y_train):
        print(len(train), len(test))
        x_train5 = x_train.iloc[train]
        y_train5 = y_train[train]
        x_test5 = x_train.iloc[test]
        y_test5 = y_train[test]
        nep_model.fit(x_train5, y_train5)
        
        accuracy=accuracy_score(y_test5, nep_model.predict(x_test5),normalize=True)
        auc=roc_auc_score(y_test5, nep_model.predict_proba(x_test5)[:,1])
        
        cvacc.append(accuracy)
        cvauc.append(auc)
    dic={'acc':cvacc,'acu':cvauc}

    # cv_solver=cross_validate(nep_model, x_train, y_train, cv=5, scoring="roc_auc",
    #                          return_estimator=True)
    
    # train_score=cv_solver['test_score']
    # print("training scores are" + str(train_score))

    # solver_list=cv_solver['estimator']
    # nep_model=solver_list[-1]
        
    filename = y_tune.name.replace(" ", "") + '_NEPLR.pkl'
    joblib.dump(nep_model, os.path.join(output,filename))
    
    filename_iv = y_tune.name.replace(" ", "") + '_NEPLR_iv.pkl'
    f=open(filename_iv,"wb")
    pickle.dump(dic,f)
    f.close()
    
    y_score=nep_model.predict_proba(x_test)[:,1]
    
    y_predict = nep_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename


# ### Naive Bayes

# In[5]:


def NB(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn.naive_bayes import GaussianNB
    nbg=GaussianNB()
    parameter_space={"var_smoothing": [1e-10, 1e-9, 1e-8]}
    
    if search == "grid":
        clf = GridSearchCV(nbg, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    else:
        clf = RandomizedSearchCV(nbg, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    clf.fit(x_tune, y_tune)
    

    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    best_score=clf.best_score_


    nb_model=GaussianNB(var_smoothing=best_params['var_smoothing'])

    cv_solver=cross_validate(nb_model, x_train, y_train, cv=5, scoring="roc_auc",
                             return_estimator=True)
    
    train_score=cv_solver['test_score']
    print("training scores are" + str(train_score))

    solver_list=cv_solver['estimator']
    nb_model=solver_list[-1]
        
    filename = y_tune.name.replace(" ", "") + '_NB.pkl'
    joblib.dump(nb_model, os.path.join(output,filename))
    
    y_score=nb_model.predict_proba(x_test)[:,1]
    
    y_predict = nb_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename


# ### Logistic Regression

# In[6]:


def LR(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn.linear_model import LogisticRegression
    
    lr_model=LogisticRegression(random_state=0)

    cv_solver=cross_validate(lr_model, x_train, y_train, cv=5, scoring="roc_auc",
                             return_estimator=True)
    
    train_score=cv_solver['test_score']
    print("training scores are" + str(train_score))
    solver_list=cv_solver['estimator']
    lr_model=solver_list[-1]
        
    filename = y_tune.name.replace(" ", "") + '_LR.pkl'
    joblib.dump(lr_model, os.path.join(output,filename))
    
    y_score=lr_model.predict_proba(x_test)[:,1]
    
    y_predict = lr_model.predict(x_test)
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename

def NEPLR(x_tune,y_tune,x_train,y_train,x_test, y_test, search):
    from sklearn.linear_model import ElasticNet
    nep=ElasticNet(max_iter=5000)
    parameter_space={"alpha": [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,1,1.5,2,2.5,3], "l1_ratio": [0.2,0.4,0.6,0.8]}
    
    if search == "grid":
        clf = GridSearchCV(nep, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    else:
        clf = RandomizedSearchCV(nep, parameter_space, scoring="roc_auc", n_jobs=-1, cv=5, verbose =1)
    clf.fit(x_tune, y_tune)
    

    print('Best parameters found:\n', clf.best_params_)
    best_params=clf.best_params_
    best_score=clf.best_score_


    enplr_model=ElasticNet(alpha=best_params['alpha'], 
                           l1_ratio=best_params['l1_ratio'],
                           max_iter=5000)

    cv_solver=cross_validate(enplr_model, x_train, y_train, cv=5, scoring="roc_auc",
                             return_estimator=True)
    
    train_score=cv_solver['test_score']
    solver_list=cv_solver['estimator']
    enplr_model=solver_list[-1]
        
    filename = y_tune.name.replace(" ", "") + '_NEPLR.pkl'
    joblib.dump(enplr_model, os.path.join(output,filename))
    
    #.predict is same as .predict_proba for other models
    y_score=enplr_model.predict(x_test)
    y_predict = y_score > 0.5
    
    print(confusion_matrix(y_test, y_predict))
    
    return y_score, filename
#______________________________________________________________________
#______________________________________________________________________

# ## Validation Plots
# ### ROC Curve

# In[9]:


def AUC(y_test, y_score, filename):
    ############# Receiver Operating Characterisitc
    # use predict_proba or decision_function to generate the data for the plot
    #saving models - change name
    #ann_model=joblib.load('REVadmin_annmodel.pkl')

    #y_test_b=label_binarize(y_test, classes=[0,1])
    #n_classes=y_test_b.shape[1]

    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #for i in range(n_classes):
    #i=0

    #calculating true positive rate and false positive rate
    fpr, tpr, thresholds = roc_curve(y_test[:], y_score, pos_label=1)
    roc_auc=roc_auc_score(y_test, y_score)
    #roc_auc = auc(fpr, tpr)
    length=fpr.shape[0]-1
    indexs=[]
    for i in range(length):
        if fpr[i]==fpr[i+1]:
            indexs.append(i)
        else:
            continue
    fpr=np.delete(fpr,indexs)  
    tpr=np.delete(tpr,indexs) 
    
    f = interp1d(fpr,tpr,kind='cubic')
    fpr_new=np.linspace(0, 1, num=100, endpoint=True)

    plt.figure(figsize=(8.7,3.9), dpi=600)
    lw = 2
    plt.plot(
        fpr_new,
        f(fpr_new),
        color="slateblue",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc)

    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Receiver operating characteristic curve",fontdict={'fontsize':11})
    plt.legend(loc="lower right")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "11"
    filename = filename.replace('.pkl', '_AUC.jpg')
    # path = os.path.join(figure,filename)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()


# ### Calibration Plot

# In[10]:


def CAL(y_test, y_score, filename):
    ###### Calibration plot
    coor_y, coor_x=calibration_curve(y_test, y_score, n_bins=20)
    brier_score=brier_score_loss(y_test, y_score)
   
    # f = interp1d(coor_x,coor_y,kind='cubic',fill_value="extrapolate")
    # coor_x_new=np.linspace(0, 0.73, num=50, endpoint=True)    

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    
    
    slop, intecept,r_value,p_value,std_err=linregress(coor_x,coor_y)
    coor_x_new=np.linspace(0, 1, num=101, endpoint=True)
    coor_y_new=intecept+slop*coor_x_new
    
    
    plt.figure(figsize=(8.7,3.9), dpi=600)
    # plt.plot(coor_x, coor_y, color="slateblue",marker='o', linewidth=1, label='Brier_Score: %0.2f' %brier_score)
    plt.scatter(coor_x, coor_y, color="slateblue",marker='o', alpha=0.8, label='Original data')
    plt.plot(coor_x_new, coor_y_new, color="orange", linewidth=2, label='Fitted line')
    plt.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label='Reference line')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title("Calibration plot",fontdict={'fontsize':11})
    plt.xlabel("Predicted value")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="lower right")
    
    filename = filename.replace('.pkl', '_CAL.jpg')
    
    plt.rcParams["font.size"] = "11"
    plt.rcParams['font.family'] = 'Times New Roman'
    # path = os.path.join(figure,filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
               
    slop, intecept,r_value,p_value,std_err=linregress(coor_x,coor_y)
    result = {"Brier Score" : brier_score, "slope" : slop, "intercept": intecept, "R_value": r_value, "P_value" : p_value, "Standar Error" : std_err}
    print("Calibration curve: " + str(result))


# ### Decision Curve

# In[11]:


def DC(y_test, y_score, filename):
    ##### Decision Curve
    net_benefit_model = np.array([])
    net_benefit_all = np.array([])
    test_ans = y_test
    
    thresh_group=[]


#for all benefit
    tna, fpa, fna, tpa = confusion_matrix(test_ans, test_ans).ravel()
    total = tpa + tna
    
    for i in range(0,1000,1):
        thresh=i/1000
        thresh_group = np.append(thresh_group, thresh)
        y_pred_label = y_score > thresh

        tn, fp, fn, tp = confusion_matrix(test_ans, y_pred_label).ravel()
        n = len(y_test)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
        
        net_benefit = (tpa / total) - (tna / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
        
    plt.figure(figsize=(8.7,3.9), dpi=600)
    plt.plot(thresh_group, net_benefit_model, color = 'slateblue', label = 'Model')
    plt.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    plt.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')
    
    
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    plt.fill_between(thresh_group, y1, y2, color = 'slateblue', alpha = 0.2)
    #Figure Configuration, Beautify the details 
    plt.xlim(0,1)
    plt.ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    plt.xlabel(
    xlabel = 'Threshold probability',
    fontdict= {
    'family': 'Times New Roman', 'fontsize': 11}
    )
    plt.ylabel(
    ylabel = 'Net benefit',
    fontdict= {
    'family': 'Times New Roman', 'fontsize': 11}
    )
    plt.title('Decision curve analysis',fontdict={'fontsize':11})
    plt.grid('major')
    # plt.spines['right'].set_color((0.8, 0.8, 0.8))
    # plt.spines['top'].set_color((0.8, 0.8, 0.8))
    plt.legend(loc = 'upper right')
    filename = filename.replace('.pkl', '_DC.jpg')
    # path = os.path.join(figure,filename)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()





# In[12]:


def allplot(y_test, y_score, filename):
    AUC(y_test, y_score, filename)
    CAL(y_test, y_score, filename)
    DC(y_test, y_score, filename)


# ## Running the Models
# ### Loading Data

# In[8]:


Feature_clean_tune=pd.read_csv('TOTHLOS_REVHIP_tune.csv')

x_tune=Feature_clean_tune.iloc[0:,0:-1]
x_tune=x_tune.iloc[0:,1:]
y_tune=Feature_clean_tune.iloc[0:,-1]

Feature_clean_train=pd.read_csv('TOTHLOS_REVHIP_train.csv')

x_train=Feature_clean_train.iloc[0:,0:-1]
x_train=x_train.iloc[0:,1:]
y_train=Feature_clean_train.iloc[0:,-1]

Feature_clean_test=pd.read_csv('TOTHLOS_REVHIP_test.csv')


x_test=Feature_clean_test.iloc[0:,0:-1]
x_test=x_test.iloc[0:,1:]
y_test=Feature_clean_test.iloc[0:,-1]

y_tune.name = y_train.name.replace("<", "")
y_train.name = y_tune.name.replace("<", "")
                    


# ### Run Each Model and Save Plots

# In[24]:


y_score7, filename7 = LR(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')

allplot(y_test, y_score7, filename7)


# In[25]:


y_score6, filename6 = NB(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')

allplot(y_test, y_score6, filename6)


# In[26]:


y_score, filename = ANN_train(ANN_model,x_train, y_train, x_test)


allplot(y_test, y_score, filename)

    
    

# In[28]:


y_score1, filename1 = RF(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')
allplot(y_test, y_score1, filename1)

y_score2, filename2 = SGB(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')

allplot(y_test, y_score2, filename2)
# In[29]:


y_scorez, filenamez = HGB(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')
allplot(y_test, y_scorez, filenamez)


# In[13]:


y_score3, filename3 = SVM(x_tune,y_tune,x_train,y_train,x_test, y_test, 'random')

allplot(y_test, y_score3, filename3)


# In[ ]:


y_score4, filename4 = KNN(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')

allplot(y_test, y_score4, filename4)


# In[ ]:


y_score5, filename5 = NEPLR(x_tune,y_tune,x_train,y_train,x_test, y_test, 'grid')

allplot(y_test, y_score5, filename5)


