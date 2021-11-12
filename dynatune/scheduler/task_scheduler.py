import time
from ..task_state import TaskState
"""
Base class of tasks scheduler which allocate the time slices

Parameters
----------
tasks: list of Task
    Tasks to schedule.
budget: float
    time budget of this schedule, seconds
slice_size: float
    Time slice size when allocating the time resource
measure_option: dict
    The options for how to measure generated code.
    You should use the return value of autotvm.measure_option for this argument.
tuner: string, optional
    Tuner to do the tuning for each task. 'xgb' for XGBTuner;
    'ga' for GATuner; 'random' for RandomTuner; 'gridsearch' for GridSearchTuner;
    'rfei' for RFEITuner. Default is 'rfei'.
early_stopping: int, optional
    Early stop the tuning when not finding better configs in this number of trials
log_filename: string, optional
    The file name to save tuning result.
n_trial: int, optional
    Maximum number of configs to try (measure on real hardware)
"""
class TaskScheduler:
    def __init__(self, 
                 tasks, 
                 budget, 
                 slice_size, 
                 measure_option,
                 tuner='rfei',
                 early_stopping=None,
                 log_filename='tuning.log',
                 n_trial=1e30):
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.budget = budget 
        self.slice_size = slice_size
        self.measure_option = measure_option
        self.tuner = tuner
        self.early_stopping = early_stopping
        print("early_stopping is", self.early_stopping)
        self.log_filename = log_filename
        
        self.states = []
        # Initialize the states for each task
        for i in range(self.n_tasks):
            tsk = tasks[i]
            n_trial = min(n_trial, len(tsk.config_space))
            self.states.append(TaskState(i+1, tsk, tuner, self.n_tasks, n_trial))
            
        self.active_tasks = [a+1 for a in range(self.n_tasks)]
        print("\nInitialize active_tasks")
        print(self.active_tasks)
    
    def pickNextTask(self):
        """Get the potential task to tune next
        Returns
        -------
        task_no: the task_no of the picked task
        """
        return NotImplementedError()
    
    def warm_up(self, min_gflops=0):
        print("Warming up...")
        for i in range(self.n_tasks):
            cur = self.states[i]
            while cur.flops_max<=min_gflops and not cur.is_finished:
                cur.tune_slice(self.slice_size, self.early_stopping, self.measure_option, self.log_filename)
    
    def schedule(self):
        """Main method of the scheduling process
        Warm up first.
        Then run one time slice in each iteration.
        """
        start = time.time()
        self.warm_up()
        while len(self.active_tasks)>0 and time.time()-start<self.budget:
            nxt_tno = self.pickNextTask()
            print("\nPick up task_no=%d"%nxt_tno)
            nxt = self.states[nxt_tno-1]
            nxt.tune_slice(self.slice_size, self.early_stopping, self.measure_option, self.log_filename)
            
            if nxt.is_finished:
                self.active_tasks.remove(nxt.task_no)
                print("\nafter remove, active_tasks:")
                print(self.active_tasks)
        print("Scheduling finished.")
