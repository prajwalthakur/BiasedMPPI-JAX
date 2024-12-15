from dataclasses import dataclass
from scipy.stats import truncnorm
import numpy as np
SEED_VALUE = 113 
np.random.seed(SEED_VALUE)
rng = np.random.RandomState(SEED_VALUE) 
import pdb
@dataclass
class turncated_normal_delay:
  low:np.float32
  high: np.float32
  mu: np.float32
  sigma: np.float32
  def __init__(self,mu:np.float32 , sigma:np.float32  , low:np.float32 , high: np.float32):
    self.low = low
    self.high = high
    self.mu = mu
    self.sigma = sigma
    self.a  = (self.low - self.mu)/self.sigma
    self.b =   (self.high - self.mu)/self.sigma
    
  def sample_delay(self,size:np.ndarray):
      samples = truncnorm.rvs(self.a, self.b, loc=self.mu, scale=self.sigma, size=size,random_state=rng)
      return samples
 
 
 
import numpy as np
from scipy.stats import truncnorm

class MultimodalTruncatedNormalDelay:
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray, low: np.float32, high: np.float32, weights: np.ndarray):
        """
        Initialize with arrays for means (mus), standard deviations (sigmas), and weights.
        `low` and `high` represent truncation boundaries.
        """
        self.low = low
        self.high = high
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        
        # Precompute the truncation values for each normal distribution
        self.a = (self.low - self.mus) / self.sigmas
        self.b = (self.high - self.mus) / self.sigmas
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(self.weights), 1), "Weights must sum to 1"
    
    def sample_delay(self, size: int, rng=np.random.default_rng()):
        """
        Sample from the multimodal truncated normal distribution.
        """
        # Choose which mode to sample from based on the weights
        components = rng.choice(len(self.mus), size=size, p=self.weights)
        
        # Initialize an array to hold the samples
        samples = np.zeros(size)
        
        # For each component, sample from the corresponding truncated normal distribution
        for i in range(len(self.mus)):
            mask = components == i
            if np.any(mask):
                samples[mask] = truncnorm.rvs(self.a[i], self.b[i], loc=self.mus[i], scale=self.sigmas[i], size=np.sum(mask), random_state=rng)
        
        return samples
 
 
 
    
@dataclass
class constant_delay:
  delay : np.float32
  
  def __init_(self, delay:np.float32):
    self.delay = delay
    
  def sample_delay(self,size:np.ndarray=0):
      samples = np.ones(size)*self.delay
      return samples
    
# import pdb
# delay_model_normal = turncated_normal_delay(mu=0.35,sigma=0.2,low=0.0,high=0.7)
# pdb.set_trace()
# delay_model_normal.sample_delay(size=1) 
# pdb.set_trace()