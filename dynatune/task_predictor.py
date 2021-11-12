from .pylrpredictor.curvemodels import MCMCCurveModel,MLCurveModel,MlCurveMixtureModel
from .pylrpredictor.curvefunctions import all_models
import numpy as np
from scipy.stats import norm
"""
The model to fit the learning curve
"""
class TaskPredictor:
    def __init__(self):
        self.model = None
        self.curmax = None
        self.MAX_CPU_GFLOPS = 200
        self.MAX_GPU_GFLOPS = 9400
        
    def fit(self, xs, ys):
        pass
    
    def predict(self, x):
        return NotImplementedError
    
    def expected_improvement(self, best_flops, mu, sigma):
        with np.errstate(divide='ignore'):
            Z = (mu - best_flops) / sigma
            ei = (mu - best_flops) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] == max(0.0, mu-best_flops)
        return ei   
    
    def predict_gain(self, tState, slice_size, use_ei=False):
        """Get the gain of the task
        
        Parameters
        ----------
        tState: TaskState
            Given all the info the prediction needs
        slice_size: int
            At how long the future to predict
        use_ei: bool, optional
            Whether to use expected improvement to calculate the gain.
        """
        pre_x = int(slice_size // (tState.time_cost/tState.iters) + tState.iters)
        pre_y, std = self.predict(pre_x)
        print("Predict pre_x:%d, pre_y:%f, std:%f" % (pre_x, pre_y, std))
        if use_ei:
            pre_y = self.expected_improvement(tState.flops_max, pre_y, std)
        return pre_y
    
class MLPredictor(TaskPredictor):
    def __init__(self, func='pow2'):
        self.curve = MLCurveModel(function=all_models[func])
        
    def fit(self, xs, ys):
        return self.curve.fit(xs, ys)
    
    def predict(self,x):
        pre = self.curve.predict(x)
        return pre, 0 # std is 0

class MCMCPredictor(TaskPredictor):
    """Predictor with mcmc simulating process
    
    Parameters
    ----------
    func: string, optional
        Function name to fit. Optional names could be seen in pylrpredictor.curvefunctions
        Default is log_power.
    nwalkers: int, optional
        Emcee parameter. The number of walkers in the ensemble.
    nsamples: int, optional
        Emcee parameter. The number of samples in the ensemble.
    burn_in: int, optional
        Emcee parameter. The number of iterations from which the walking turns into stable.
    nthreads: int, optional
        Emcee parameter. The number of threads.
    """
    def __init__(self, func='log_power', nwalkers=10, nsamples=500, burn_in=300, nthreads=1):
        self.curve = MCMCCurveModel(function=all_models[func], nwalkers=nwalkers, nsamples=nsamples, burn_in=burn_in, nthreads=nthreads)
        
    def fit(self, xs, ys):
        return self.curve.fit(xs, ys)
        
    def predict(self, x):
        pre, std = self.curve.predict(x)
        return pre, std