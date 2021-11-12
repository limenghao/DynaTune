from .task_scheduler import TaskScheduler
from ..task_predictor import MLPredictor, MCMCPredictor
from ..task_selector import UCB1Selector
import math

class MultiArmBanditScheduler(TaskScheduler):
    """ The scheduler could use predictor and selector 
    to schedule the tuning tasks dynamically
    
    Parameters
    ----------
    tasks: list of Task
         The tasks to schedule
    budget: int
         Time budget, seconds
    slice_size: int
         The unit of time to schedule the time budget, seconds
    measure_option: dict
         The options for how to measure generated code.
         You should use the return value of autotvm.measure_option for this argument.
    tuner: string, optional
         Tuner type: xgb, random, gridsearch, ga, rf,...
    early_stopping: int, optional
         Early stop the tuning when not finding better configs in this number of trials
    predictor: string, optional
         The predictor used to predict gain of each task
    selector: string, optional
         The selector used to select one task based on all the gains
    """
    def __init__(self,
                 tasks, 
                 budget, 
                 slice_size, 
                 measure_option,
                 tuner='xgb',
                 early_stopping=None,
                 log_filename='tuning.log',
                 predictor="mcmc",
                 selector="ucb1"):
        super(MultiArmBanditScheduler, self).__init__(tasks,budget,slice_size,measure_option,tuner,early_stopping,log_filename)
        if predictor=='ml':
            self.predictor=MLPredictor('pow2')
        elif predictor=='mcmc':
            self.predictor=MCMCPredictor(func='log_power', nwalkers=10, nsamples=500, burn_in=300, nthreads=1)
        else:
            raise Exception("Only support ml,mcmc predictor!")
        if selector=="ucb1":
            self.selector = UCB1Selector(c=2)
        else:
            raise Exception("Only supoort ucb1 selector!")
    
    def pickNextTask(self):
        scores = []
        costs = []
        for tsk_no in self.active_tasks:
            tState = self.states[tsk_no-1]
            # Only re-calculate the task score when the task got a time slice and has update
            if tState.update:
                self.predictor.fit(tState.xs, tState.ys)
                tscore = self.predictor.predict_gain(tState, self.slice_size, use_ei=False)
                tState.score = tscore
                tState.update = False
                print("\n Get tscore for tsk#%d is %f" % (tsk_no, tscore))
            scores.append(tState.score)
            costs.append(tState.slices)
        nxt = self.selector.select(self.active_tasks, scores, costs)
        return nxt
        