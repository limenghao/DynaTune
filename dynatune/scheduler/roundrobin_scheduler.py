from .task_scheduler import TaskScheduler

class RoundRobinScheduler(TaskScheduler):
    """Schedule the tasks tuning in a sequential order.
    """
    def __init__(self,
                 tasks, 
                 budget, 
                 slice_size, 
                 measure_option,
                 tuner='xgb',
                 early_stopping=None,
                 log_filename='tuning.log'):
        super(RoundRobinScheduler, self).__init__(tasks,budget,slice_size,measure_option,tuner,early_stopping,log_filename)
        self.cur_tno = 0 #current task_no
    
    def pickNextTask(self):
        self.cur_tno += 1
        if self.cur_tno>self.active_tasks[-1]:
            self.cur_tno = self.active_tasks[0]
            return self.cur_tno
        while self.cur_tno not in self.active_tasks:
            self.cur_tno += 1
        return self.cur_tno