from .task_scheduler import TaskScheduler
import random

class RandomTaskScheduler(TaskScheduler):
    """Schedule the tasks tuning randomly.
    """
    def __init__(self,
                 tasks, 
                 budget, 
                 slice_size, 
                 measure_option,
                 tuner='xgb',
                 early_stopping=None,
                 log_filename='tuning.log',
                 rseed=0):
        super(RandomTaskScheduler, self).__init__(tasks,budget,slice_size,measure_option,tuner,early_stopping,log_filename)
        random.seed(rseed)
    
    def pickNextTask(self):
        # generate a random integer in the range of [0, length_of_active_tasks-1]
        god = random.randint(0,len(self.active_tasks)-1) #index
        return self.active_tasks[god] #task_no