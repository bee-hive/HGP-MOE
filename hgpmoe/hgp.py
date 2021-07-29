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

class HierarchicalGP(gpflow.models.BayesianModel, InternalDataTrainingLossMixin):
    def __init__(self, 
                 data, 
                 cluster_assignments,
                 taper=False,
                 noise_variance=1.0, 
                 mean_function=None, 
                 name='HierarchicalGP'):
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
        self.G = tf.unique(self.C_assignments)[0].numpy()
        self.G = np.sort(self.G)
        self.noise = self.G.shape[0] * [0.001] 
        self.K_group_list = []
        for g in self.G:
            if taper:
                self.K_group_list.append(TaperedMatern(kern_type="Matern52", 
                                                       name='group_kernel%d'%int(g)))
            else:
                self.K_group_list.append(gpflow.kernels.Matern52(name='group_kernel%d'%int(g)))
            
            cluster_idx, = np.where(self.C_assignments == g)
            # For each patient in the cluster compute distance to every other patient
            Y_subset = [self.Y[j] for j in cluster_idx]
            cluster_distances = self.Wasserstein.K(Y_subset)                        
            # Set distance minimizer as cluster representative
            minimizer = np.argmin(np.sum(cluster_distances, axis=1))
            print("Patient ", cluster_idx[minimizer], " minimizes cluster ", g)
        
            
        # Patient level kernels
        self.K_patient_list = []
        for i in range(len(self.X)):
            g = self.C_assignments[i]
            g_idx, = np.where(self.G == g)
            g_idx = g_idx[0]
            self.K_patient_list.append(gpflow.kernels.Matern32(name='patient_kernel%d'%i))
            
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(noise_variance)
        
        self.NLL_sum = [None] * len(self.K_group_list)
            

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
            
        if (patient_idx is not None) and (None not in self.NLL_sum):
            g = self.C_assignments[patient_idx]
            cluster_idx, = np.where(self.C_assignments == g)
            g_idx, = np.where(self.G == g)
            g_idx = g_idx[0]
            blocks = []
            Xg = []
            Yg = []
            for i in cluster_idx:
                # Xi = tf.gather(self.X, i, 0)
                Xi = self.X[i]
                Xi = tf.expand_dims(Xi, axis=-1)
                Yi = tf.expand_dims(self.Y[i], axis=-1)
                Xg.append(Xi)
                Yg.append(Yi)
                blocks.append(self.K_patient_list[i](Xi))
                    
            Xg = tf.concat(Xg, 0)
            Yg = tf.concat(Yg, 0)
            linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
            linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
            Kfs = linop_block_diagonal.to_dense()
            Kg = self.K_group_list[g_idx](Xg)


            ks = self._add_noise_cov(Kfs + Kg)
            L = tf.linalg.cholesky(ks)
            m = self.mean_function(Xg)
            log_prob = multivariate_normal(Yg, m, L)
            self.NLL_sum[g_idx] = tf.reduce_sum(log_prob)        
        else:
            for g in self.G:
                cluster_idx, = np.where(self.C_assignments == g)
                g_idx, = np.where(self.G == g)
                g_idx = g_idx[0]
                blocks = []
                Xg = []
                Yg = []
                for i in cluster_idx:
                    Xi = self.X[i]
                    Xi = tf.expand_dims(Xi, axis=-1)
                    Yi = tf.expand_dims(self.Y[i], axis=-1)
                    Xg.append(Xi)
                    Yg.append(Yi)
                    blocks.append(self.K_patient_list[i](Xi))
                    
                Xg = tf.concat(Xg, 0)
                Yg = tf.concat(Yg, 0)
                linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in blocks]
                linop_block_diagonal = tf.linalg.LinearOperatorBlockDiag(linop_blocks)
                Kfs = linop_block_diagonal.to_dense()
                Kg = self.K_group_list[g_idx](Xg)
                
                
                ks = self._add_noise_cov(Kfs + Kg)
                L = tf.linalg.cholesky(ks)
                m = self.mean_function(Xg)

                log_prob = multivariate_normal(Yg, m, L)
                self.NLL_sum[g_idx] = tf.reduce_sum(log_prob)
        NLL = tf.add_n(self.NLL_sum)
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
        # Xi = tf.gather(self.X, patient_idx, 0)
        Xi = self.X[patient_idx]
        Xi = tf.expand_dims(Xi, axis=-1)
        Yi = tf.expand_dims(self.Y[patient_idx], axis=-1)
        err = Yi - self.mean_function(Xi)
        
        g = self.C_assignments[patient_idx]
        g_idx, = np.where(self.G == g)
        g_idx = g_idx[0]
        kmm = self.K_patient_list[patient_idx](Xi) + self.K_group_list[g_idx](Xi)
        knn = self.K_patient_list[patient_idx](Xnew, full_cov=full_cov) + \
                    self.K_group_list[g_idx](Xnew, full_cov=full_cov)
        kmn = self.K_patient_list[patient_idx](Xi, Xnew) + self.K_group_list[g_idx](Xi, Xnew)
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