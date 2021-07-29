from typing import Optional, Tuple

import gpflow
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from .kernels import WassersteinStationary, TaperedMatern

from gpflow.kernels import Kernel
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.models import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.conditionals.util import sample_mvn
from gpflow import set_trainable
import tensorflow_probability as tfp

class ExpertsMixGP(gpflow.models.BayesianModel, InternalDataTrainingLossMixin):
    def __init__(self, 
                 data, 
                 cluster_assignments, # dictionary of hierarchy
                 taper=False,
                 noise_variance=1.0, 
                 mean_function=None, 
                 name='ExpertsMixGP'):
        super().__init__()
        
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self.X, self.Y = data
        for j in range(len(self.X)):
            self.X[j] = data_input_to_tensor(self.X[j])
            self.Y[j] = data_input_to_tensor(self.Y[j])
        
        # Cluster Representatives kernel
        self.Wasserstein = WassersteinStationary(name='Kg')
        set_trainable(self.Wasserstein.variance, False)
        set_trainable(self.Wasserstein.lengthscales, False)
        
        # Number of groups
        self.C_assignments = cluster_assignments 
        all_vals = []
        for val in cluster_assignments.values():
            all_vals += val
        self.G = set(all_vals)
        self.K_group = dict()
        for g in self.G:
            if taper:
                self.K_group[g] = TaperedMatern(kern_type="Matern52", 
                                                   name='group_kernel%d'%int(g))
            else:
                self.K_group[g] = gpflow.kernels.Matern52(name='group_kernel%d'%int(g))
            
            cluster_idx = [k for k, v in self.C_assignments.items() if g in v] 
            # For each patient in the cluster compute distance to every other patient
            Y_subset = [self.Y[j] for j in cluster_idx]
            cluster_distances = self.Wasserstein.K(Y_subset)                        
            # Set distance minimizer as cluster representative
            minimizer = np.argmin(np.sum(cluster_distances, axis=1))
            print("Patient ", cluster_idx[minimizer], " minimizes cluster with attribute", g)
        
            
        # Patient level kernels
        self.K_patient = dict()
        for i in range(len(self.X)):
            self.K_patient[i] = gpflow.kernels.Matern52(name='patient_kernel%d'%i)
            
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(noise_variance)            

    def maximum_log_likelihood_objective(self, patient_idx=None):
        return self.log_marginal_likelihood(patient_idx=patient_idx)

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill(tf.shape(k_diag), self.likelihood.variance)
        return tf.linalg.set_diag(K, k_diag + s_diag)

    def log_marginal_likelihood(self, patient_idx=None):
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        Kgs = []
        Yg = tf.concat([tf.expand_dims(self.Y[i], axis=-1) for i in self.K_patient.keys()], 0)
        Xg = tf.concat([tf.expand_dims(self.X[i], axis=-1) for i in self.K_patient.keys()], 0)
        
        blocks = [self.K_patient[i](tf.expand_dims(self.X[i], axis=-1)) for i in self.K_patient.keys()]
        linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
        linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
        Kf = linop_block_diagonal.to_dense()
            
        for g in self.K_group.keys():
            cluster_idx = [k for k, v in self.C_assignments.items() if g in v] 
            KXg = self.K_group[g](Xg)
            mask = np.ones(KXg.shape)
            index = 0
            for i in self.K_patient.keys():
                if i not in cluster_idx:
                    mask[index:index + len(self.X[i]), :] = 0
                    mask[:, index:index + len(self.X[i])] = 0
                index += len(self.X[i])
            mask = tf.constant(mask)
            Kgs.append(tf.math.multiply(KXg, mask))

        Kg = tf.math.add_n(Kgs)
        ks = self._add_noise_cov(Kf + Kg)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(Xg)

        log_prob = multivariate_normal(Yg, m, L)
        NLL = tf.reduce_sum(log_prob)        
        return NLL
    
    def predict_y(self, Xnew, patient_idx, full_cov=False, full_output_cov=False):
        """
        Compute the mean and variance of the held-out data at the input points.
        """
        f_mean, f_var = self.predict_f(Xnew, patient_idx, full_cov=full_cov, 
                                       full_output_cov=full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_f(self, Xnew, patient_idx, full_cov=False, full_output_cov=False):
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        Xi = self.X[patient_idx]
        Xi = tf.expand_dims(Xi, axis=-1)
        Yi = tf.expand_dims(self.Y[patient_idx], axis=-1)
        err = Yi - self.mean_function(Xi)
        
        g = self.C_assignments[patient_idx]

        kmm = [self.K_patient[patient_idx](Xi)]
        knn = [self.K_patient[patient_idx](Xnew, full_cov=full_cov)]
        kmn = [self.K_patient[patient_idx](Xi, Xnew)]
        for g in self.C_assignments[patient_idx]:
            kmm.append(self.K_group[g](Xi))
            knn.append(self.K_group[g](Xnew, full_cov=full_cov))
            kmn.append(self.K_group[g](Xi, Xnew))
            
        kmm = tf.math.add_n(kmm)
        knn = tf.math.add_n(knn)
        kmn = tf.math.add_n(kmn)
        kmm_plus_s = self._add_noise_cov(kmm)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        ) 
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
    def predict_f_samples(self, Xnew, num_samples, patient_idx, full_cov=False, full_output_cov=False):
        mean, cov = self.predict_f(Xnew, patient_idx, full_cov=full_cov, 
                                   full_output_cov=full_output_cov)
        if full_cov:

            mean_for_sample = tf.linalg.adjoint(mean)  
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=d
            )  
            samples = tf.linalg.adjoint(samples) 
        else:

            samples = sample_mvn(
                tf.expand_dims(mean, -1), tf.expand_dims(cov, -1), full_output_cov, num_samples=num_samples
            )
        return samples 
    
    def predict_log_density(self, data, patient_idx, full_cov=False, full_output_cov=False):
        """
        Compute the log density of the data at the new data points.
        """
        X, Y = data
        f_mean, f_var = self.predict_f(X, patient_idx, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.predict_log_density(f_mean, f_var, Y)