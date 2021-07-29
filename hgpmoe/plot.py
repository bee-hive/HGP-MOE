import gpflow
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
import os
import operator

plt.style.use("ggplot")
warnings.filterwarnings('ignore')
np.random.seed(0)

def pred_x(model, patient_idx, X, Y, 
           cluster_assignments,
           X_test=None, Y_test=None, 
           model_name='', n_test_points=20, feature='', seed=1):
    '''Generate test points for prediction for HGP/MOE Models'''
    color = 'blue'

    xx = np.linspace(0, np.max(X[patient_idx]), 100)[:, None]

    ## predict mean and variance of latent GP at test points
    if model_name == 'GPR':
        mean_x, var_x = model.predict_y(xx, full_cov=False, full_output_cov=False)
    else:
        mean_x, var_x = model.predict_y(xx, patient_idx)
        
    ## generate samples from posterior
    if X_test is not None:
        xnew = np.array(X_test[patient_idx])[:, None]
        xnew = xnew.astype(np.float64)
    else:
        xnew = np.linspace(np.max(X[patient_idx]), 
                           np.max(X[patient_idx]) + n_test_points, n_test_points)[:, None]
        
    if model_name == 'GPR':
        pred_mean, pred_var = model.predict_y(xnew, full_cov=False, full_output_cov=False)
    else:
        pred_mean, pred_var = model.predict_y(xnew, patient_idx)
    
    ## Compute MSE
    x_train = np.array(X[patient_idx])[:, None]
    x_train = x_train.astype(np.float64)
    if model_name == 'GPR':
        train_pred_mean, train_pred_var = model.predict_y(x_train, full_cov=False, full_output_cov=False)
    else:
        train_pred_mean, train_pred_var = model.predict_y(x_train, patient_idx)
    y_train = Y[patient_idx]
    train_error = np.square(np.subtract(y_train, train_pred_mean.numpy().flatten())).mean()
        
    test_error = 0
    if X_test is not None:
        x_test = np.array(X_test[patient_idx])[:, None]
        x_test = x_test.astype(np.float64)
        if model_name == 'GPR':
            test_pred_mean, test_pred_var = model.predict_y(x_test, full_cov=False, full_output_cov=False)
        else:
            test_pred_mean, test_pred_var = model.predict_y(x_test, patient_idx)
        y_test = Y_test[patient_idx]
        test_error = np.square(np.subtract(y_test, test_pred_mean.numpy().flatten())).mean()

    ## plot
    fig = plt.figure(figsize=(12, 6))
    plt.errorbar(X[patient_idx], Y[patient_idx], 
             yerr=0.1, color='black', 
                 capsize=3, elinewidth=1, fmt='o',
                label='noisy observations')
    plt.plot(xx, mean_x, color=color, lw=2, label='mean function')
    plt.fill_between(
        xx[:, 0],
        mean_x.numpy().flatten() - 1.96 * np.sqrt(np.abs(var_x.numpy().flatten())),
        mean_x.numpy().flatten() + 1.96 * np.sqrt(np.abs(var_x.numpy().flatten())),
        color=color,
        alpha=0.2, 
        label='fitted variance'
    )

    plt.plot(xnew, pred_mean, "d", color='blue', label='predicted values')

    if Y_test is not None:
        y_test = Y_test[patient_idx]
        plt.plot(X_test[patient_idx], y_test, 'o',
                    color='red', label='test points')

    if model_name == 'MOE':
        str_title = model_name + '\n' + \
                'Patient %d ' %patient_idx + feature + \
                " Group " + ', '.join(str(e) for e in cluster_assignments[patient_idx])
    else:
        str_title = model_name + '\n' + \
                'Patient %d ' %patient_idx + feature + \
                " Group %d" %cluster_assignments[patient_idx]
    
    plt.title(str_title + "\n Train MSE %.5f | Test MSE %.5f" %(train_error, test_error))
    plt.ylabel(feature + ' measurement')
    plt.xlabel('time since admission (scaled)')
    plt.ylabel(feature + ' measurement')

    handles, labels = plt.gca().get_legend_handles_labels()
    hl = sorted(zip(handles, labels),
                key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)

    plt.legend(handles2, labels2)
    plt.show()
