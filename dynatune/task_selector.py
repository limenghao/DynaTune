import math

class TaskSelector:
    """
    Root class of selector, which defines the strategy 
    to select the task to tune among tasks
    """
    def __init__(self):
        pass
    
    def select(self, tasks, scores, costs):
        """
        Select the task among the tasks with their calculated scores
        
        Parameters
        ----------
        tasks: list of int
            list of active task_nos
        scores: list of float
            list of active tasks' scores
        costs: list of float
            list of active tasks' slices so far
        """
        raise NotImplementedError()
        
        
class UCB1Selector(TaskSelector):
    """ TaskSelector based on UCB1 formula
        Formula: Ut(a) = argmax a ∈ {1,…,K} Q(a) + c * sqrt(𝑙𝑜𝑔𝑡/(𝑁𝑡(𝑎)))
         All tasks use the same c
         Q(a) is the predicted gain given by curve model(s)
         t is the total time slices
         Nt(a) is the time slices of task a
    
    Parameters
    ----------
    c: float
        The argument in the formula c, which decides the dicount of item2, 
        default value is 2.
    """
    def __init__(self, c=2):
        self.c = c
        super(UCB1Selector, self).__init__()
        
    def select(self, tasks, scores, costs):
        """
        Use the UCB1 formula to select.
        
        Parameters
        ----------
        tasks: list of int
            list of active task_nos
        scores: list of float
            list of active tasks' scores
        costs: list of float
            list of active tasks' slices so far
        Return
        ------
        res_tno: the selected task no of TaskState.
        """
        total_T = sum(costs)
        max_ucb = 0
        res_tno = tasks[0]
        for idx in range(len(tasks)):
            tno = tasks[idx]
            tslices = costs[idx]
            item1 = scores[idx]
            item2 = math.sqrt(self.c * math.log(total_T) / tslices)
            ucb = item1 + item2
            print([tno, item1, item2, ucb])
            if ucb > max_ucb:
                max_ucb = ucb
                res_tno = tno
        return res_tno
                