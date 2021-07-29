import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from sklearn.metrics import r2_score

def compute_metrics_gpr(models, X, Y, X_test, Y_test):
    '''Computes Train/Test MSE averaged across patients, and 
    Train/Test R^2 and CI per patient for GPR Models'''
    train_mse = []
    test_mse = []
    train_r2 = []
    test_r2 = []
    CI_count = []

    for patient_idx in range(len(X)):
        x_train = np.array(X[patient_idx])[:, None]
        x_train = x_train.astype(np.float64)
        train_pred_mean, train_pred_var = models[patient_idx].predict_y(x_train, full_cov=False, full_output_cov=False)
        y_train = Y[patient_idx]

        train_error = np.square(np.subtract(y_train, train_pred_mean.numpy().flatten())).mean()
        train_mse.append(train_error)
        train_r2.append(r2_score(y_train, train_pred_mean.numpy().flatten()))

        x_test = np.array(X_test[patient_idx])[:, None]
        x_test = x_test.astype(np.float64)
        test_pred_mean, test_pred_var = models[patient_idx].predict_y(x_test, full_cov=False, full_output_cov=False)
        y_test = Y_test[patient_idx]
        
        CI_max = test_pred_mean.numpy().flatten() + 1.96 * np.sqrt(np.abs(test_pred_var.numpy().flatten()))
        CI_min = test_pred_mean.numpy().flatten() -1.96 * np.sqrt(np.abs(test_pred_var.numpy().flatten()))
        ci = np.sum(y_test > CI_max)
        ci += np.sum(y_test < CI_min)
        CI_count.append(ci/len(y_test))
        
        test_error = np.square(np.subtract(y_test, test_pred_mean.numpy().flatten())).mean()
        test_mse.append(test_error)
        test_r2.append(r2_score(y_test, test_pred_mean.numpy().flatten()))

    return np.mean(train_mse), np.mean(test_mse), np.array(train_r2), np.array(test_r2), np.array(CI_count) * 100


def compute_metrics_hgpmoe(model, X, Y, X_test, Y_test):
    '''Computes Train/Test MSE averaged across patients, and 
    Train/Test R^2 and CI per patient for HGP and MOE Models'''
    train_mse = []
    test_mse = []
    train_r2 = []
    test_r2 = []
    CI_count = []

    for patient_idx in range(len(X)):
        x_train = np.array(X[patient_idx])[:, None]
        x_train = x_train.astype(np.float64)
        train_pred_mean, train_pred_var = model.predict_y(x_train, patient_idx)
        y_train = Y[patient_idx]

        train_error = np.square(np.subtract(y_train, train_pred_mean.numpy().flatten())).mean()
        train_mse.append(train_error)
        train_r2.append(r2_score(y_train, train_pred_mean.numpy().flatten()))

        x_test = np.array(X_test[patient_idx])[:, None]
        x_test = x_test.astype(np.float64)
        test_pred_mean, test_pred_var = model.predict_y(x_test, patient_idx)
        y_test = Y_test[patient_idx]
        
        CI_max = test_pred_mean.numpy().flatten() + 1.96 * np.sqrt(np.abs(test_pred_var.numpy().flatten()))
        CI_min = test_pred_mean.numpy().flatten() -1.96 * np.sqrt(np.abs(test_pred_var.numpy().flatten()))
        ci = np.sum(y_test > CI_max)
        ci += np.sum(y_test < CI_min)
        CI_count.append(ci/len(y_test))
        
        test_error = np.square(np.subtract(y_test,  test_pred_mean.numpy().flatten())).mean()
        test_mse.append(test_error)
        test_r2.append(r2_score(y_test, test_pred_mean.numpy().flatten()))

    return np.mean(train_mse), np.mean(test_mse), np.array(train_r2), np.array(test_r2), np.array(CI_count) * 100
