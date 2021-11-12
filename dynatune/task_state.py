from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner, RFEITuner
import time

class TaskState:
    """Main enitity class for scheduling
    
    Parameters
    ----------
    task_no: int
        Identifier number of each task in one model.
    task: Task
        The task object
    tuner: string
        Tuner string.
    n_tasks: int
        The number of tasks in the model.
    n_trial: int
        Maximum number of configs to try (measure on real hardware)
    """
    def __init__(self, task_no, task, tuner, n_tasks, n_trial):
        self.n_tasks = n_tasks
        self.task_no = task_no
        self.task = task
        self.target = task.target
        self.slices = 0
        self.iters = 0
        self.time_cost = 0
        self.flops_max = 0
        self.is_finished = False
        self.prefix = "[Task %2d/%2d] " % (task_no, n_tasks)
        self.n_trial = n_trial
        self.tuner = self.getTuner(tuner,task)
        
        # xs and ys save data for predictor
        self.xs = []
        self.ys = []
        self.update = False #only recalculate the gain when one task is updated 
    
    def getTuner(self, tuner, task):
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank',plan_size=32)
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        elif tuner == 'rfei':
            tuner_obj = RFEITuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        return tuner_obj
    
    def tune_slice(self, slice_size, early_stopping, measure_option, log_filename):
        self.update = True
        # print("Tune a slice %ds for Task#%d"%(slice_size, self.task_no))
        ss = time.time()
        n_trial = len(self.task.config_space)
        # print("cur best is %f, cur xs len is %d" % (self.flops_max, len(self.tuner.xs)))
        res_no = self.tuner.tune(
            n_trial=n_trial,
            time_budget = slice_size,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.update_task_state(self),
                autotvm.callback.dyn_progress_bar(self),
                autotvm.callback.log_to_file(log_filename)])
        ee = time.time()
        self.slices += 1
        self.time_cost += ee-ss
        if res_no == 0 or res_no == 2:
            self.is_finished = True
            print("Task#%d done." % self.task_no)
        
        print("\nRun one slice for task#%d, flops_max update as:%f" %(self.task_no, self.flops_max))
        