import numpy as np
from scipy.stats import wasserstein_distance

import gpflow
from gpflow.utilities.ops import square_distance

import tensorflow as tf
tf.config.run_functions_eagerly(True)

class WassersteinStationary(gpflow.kernels.Stationary):
    " Matern32 with wasserstein distance"

    def K(self, X, X2=None):
        r2 = self.scaled_squared_wasserstein_dist(X, X2)
        return self.K_r2(r2)
    
    def K_r2(self, r2):
        if hasattr(self, "K_r"):
            r = tf.sqrt(tf.maximum(r2, 1e-36))
            return self.K_r(r)
        raise NotImplementedError

    def K_r(self, r):
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)
    
    def scaled_squared_wasserstein_dist(self, X, X2=None):
        """
        Returns ‖d(X, X2)‖² / ℓ², i.e. the squared wasserstein distance.
        """
        if type(X) == list:
            D = len(X)
        else:
            D = X.shape[0]
    
        K_mat = [[tf.numpy_function(wasserstein_distance, 
                     [self.scale(X[i]), self.scale(X[j])], tf.float32) for j in range(D)] for i in range(D)]
        return tf.convert_to_tensor(K_mat)
        
class TaperedMatern(gpflow.kernels.Stationary):
    def __init__(self, 
                 kern_type="Matern12",
                 taper_threshold = 8.0,
                 variance=1.0, 
                 lengthscales=1.0, 
                 alpha=1.0, 
                 active_dims=None,
                 **kwargs):
        super().__init__(variance=variance, lengthscales=lengthscales, active_dims=active_dims)
        self.taper_threshold = taper_threshold
        
        # Default is Matern12
        self.sqrt_factor = np.sqrt(1.0)
        if kern_type == "Matern32":
            self.sqrt_factor = np.sqrt(3.0)
        elif kern_type == "Matern52":
            self.sqrt_factor = np.sqrt(5.0)
        
    def K(self, X, X2=None):
        r2 = self.scaled_squared_euclid_dist(X, X2)
        K_full = self.K_r2(r2)
        K_taper = tf.where(r2 > self.taper_threshold, 0.0, K_full)
        return K_taper
    
    def K_r2(self, r2):
        r = tf.sqrt(tf.maximum(r2, 1e-36))
        return self.K_r(r)
    
    def K_r(self, r):
        return self.variance * (1.0 + self.sqrt_factor * r) * tf.exp(-self.sqrt_factor * r)
    
    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Returns ‖(X - X2ᵀ) / ℓ‖², i.e. the squared L₂-norm.
        """
        return square_distance(self.scale(X), self.scale(X2))