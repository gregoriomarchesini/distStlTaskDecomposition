import numpy as np
import casadi as ca
from   itertools import chain,combinations
import networkx as nx 
import logging
from typing import Iterable, TypeVar
import matplotlib.pyplot as plt
from tqdm import tqdm
import contextlib
import io
from dataclasses import dataclass


from ..stl.graphs import *
from ..stl.stl import * 

from  .transport import Publisher



# Create a reusable StringIO buffer
f = io.StringIO()

def edge_set_from_path(path:list[int]) -> list[(int,int)] :
    """Returns the set of edges returning a path. Namley it gathers each couple of consective elements in a list as edges in the path

    Returns:
        path : path of agents
    """
    
  
    edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
    
        
    return edges


V = TypeVar("V")
def powerset(iterable : Iterable[V]) -> list[set[V]]:
    """
    Computes the power set of a set fo elements
    
    Example:
        >>> powerset([1, 2, 3])
        [ (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    power_set = chain.from_iterable(combinations(iterable, r) for r in range(1,len(iterable)+1))
    power_set = [set(x) for x in power_set] # use set operations
    return power_set


class TaskOptiContainer :
    """
        The task container is an abstraction used to store tasks to be decomposed. The main point of the task constainers is that it manages optimization variables 
       required to decompose a specific task within a specific edge
    """
    
    def __init__(self, task : StlTask) -> None:
        """
        Args :
            task : Stl task (Only tasks with collaborative predicates are accepted)
        """    
         
         
        
        if not isinstance(task.predicate,CollaborativePredicate):
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
        
        
        self._task = task
        # These attributes are enables only when the task is parametric
        self._parent_task            :"StlTask"            = None  # instance of the parent class
        self._decomposition_path     :list[int]            = []    # path of the decomposition of the task when the task is parametric
        self._neighbour_edges_id     :list[int]            = []    # list of the edges neighbouring to the task edge and expressed as a single id. Ex : (4,1)-> (41) 
        
        self._is_initialized = False
        # Create variables for optimization (if the task is not parametric they are left to None).
        self._center_var     = None 
        self._scale_var      = None 
        self._eta_var        = None 
                     
                                             
        self._coupling_constraint_size = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        
        # For each task you have to save the following informations :
        
        # lambda_{ij} -> your consensus parameter for the neighbour j.
        # lambda_{ji} -> the consensus parameter of the neighbour j for you and that is shared with you.
        # \mu_{i}     -> the lagrangian multiplier associated to this task and that you use to update the consensus variable.
        # \mu_{ji}    -> The lagrangian multipliers sent from each neighbour to you and corresponding to this task.
        
        self._consensus_param_neighbours_from_neighbours   : dict[int,np.ndarray]= {} # lambda_{ji} .
        self._consensus_param_neighbours_from_self         : dict[int,np.ndarray]= {} # lambda_{ij}.
        self._lagrangian_param_neighbours_from_neighbours  : dict[int,np.ndarray]= {} # \mu_{ji}
        self._lagrangian_param_neighbours_from_self        = np.zeros((self._coupling_constraint_size,1)) # \mu_{i}
        
        # Define shared constraint (will be used to retrive the lagrangian multipliers for it after the solution is found)
        self._task_constraint             : ca.MX = None # this the constraint that each agent has set for the particular task. This is useful to geth the langrangian multiplier
        self._average_consensus_param     : ca.MX = None # average consensus parameter for the concesusn variable y
        
    
    @property
    def has_parent_task(self):
        return  self._parent_task != None
    
    @property
    def parent_task_id(self):
        if self._parent_task is None :
            raise ValueError("The task does not have a parent task specified. If the task is parametric you can specify the parent task and the decomposition path using the set_parent_task_and_decomposition_path method")
        return self._parent_task.task_id
    
    @property
    def  parent_task(self):
        if self._parent_task is None :
            raise ValueError("The task does not have a parent task specified. If the task is parametric you can specify the parent task and the decomposition path using the set_parent_task_and_decomposition_path method")
        return self._parent_task
    
    @property
    def length_decomposition_path(self):
        if len(self._decomposition_path) == 0 :
            raise ValueError("The task does not have a decomposition path specified. If the task is parametric you can specify the parent task and the decomposition path using the set_parent_task_and_decomposition_path method")
        return len(self._decomposition_path)
    
    @property
    def neighbour_edges_id(self):
        if len(self._decomposition_path) == 0 :
            raise ValueError("The task does not have a decomposition path specified. If the task is parametric you can specify the parent task and the decomposition path using the set_parent_task_and_decomposition_path method")
        return self._neighbour_edges_id
    
    @property
    def center_var(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provided for the container")
        return self._center_var
    
    @property
    def scale_var(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provided for the container")
        return self._scale_var
    
    @property
    def eta_var(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provided for the container")
        return self._eta_var
    
    @property
    def task(self):
        return self._task
    
    @property
    def local_constraint(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if self._task_constraint != None :
            return self._task_constraint
        else :
            raise RuntimeError("local constraint was not set. call the method ""set_task_constraint_expression"" in order to set a local constraint")
    
    @property
    def consensus_param_neighbours_from_neighbours(self):
        return self._consensus_param_neighbours_from_neighbours
    
    @property
    def consensus_param_neighbours_from_self(self):
        return self._consensus_param_neighbours_from_self
    
    @property
    def average_consensus_param(self):
        return self._average_consensus_param
    
    
    def set_optimization_variables(self,opti:ca.Opti) -> None:
        """Set the optimization variables for the task container. Casadi Is used as Backend for the optimization variables"""
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric hence the optimization variables cannot be set")
        
        self._center_var   = opti.variable(self.task.predicate.state_space_dim,1)                     # center the parameteric formula
        self._scale_var    = opti.variable(1)                                                         # scale for the parametric formula
        self._eta_var     = ca.vertcat(self._center_var,self._scale_var)                                    # optimization variable
        
        coupling_constraint_size      = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        self._average_consensus_param = opti.parameter(coupling_constraint_size,1) # average consensus parameter for the concesusn variable y
        self._is_initialized          = True

    
    def compute_consensus_average_parameter(self) -> np.ndarray :
        """
        Given this agent is agent i, then the function computes sum_{j} \lambda_{ij} - \lambda_{ji} where lambda_{ji} are the current concensnsu variables from the neighbours of this agent  
        """
        average = 0
        for edge_id in self._neighbour_edges_id:
            average += self._consensus_param_neighbours_from_self[edge_id] - self._consensus_param_neighbours_from_neighbours[edge_id]
            
        return average
    
    def set_parent_task_and_decomposition_path(self,parent_task:"StlTask", decomposition_path:list[tuple[int,int]])-> None:
        """ For a give parameteric task we record a unique id associated with the paranet task from which the parameteric task is derived."""
        
        if not self._task.predicate.is_parametric :
            raise Warning("The task is not parametric. The parent task and decomposition path setting will be ignored")
        elif parent_task.is_parametric :
            raise ValueError("The parent task must not be parametric to be valid.")
        
        self._parent_task = parent_task
        self._decomposition_path = decomposition_path
        
        
        for i,j in edge_set_from_path(decomposition_path) :
            is_connected_to_this_edge = ( (i == self.task.predicate.source_agent) or (j == self.task.predicate.source_agent) or (i == self.task.predicate.target_agent) or (j == self.task.predicate.target_agent) )
            task_edge_id = edge_to_int((self.task.predicate.source_agent,self.task.predicate.target_agent))
            
            if is_connected_to_this_edge  and  (edge_to_int((i,j)) != task_edge_id) :
                self._neighbour_edges_id.append(edge_to_int((i,j)))
            
        if not len(self._neighbour_edges_id) in [1,2] :
            raise Exception(f"There is a probable implementation error for the task with id " + str(self.task.task_id) + " and decomposition path " + str(self._decomposition_path) +
                            f". The number of edge neighbours along the path is {len(self._neighbour_edges_id)} and not 1 or 2 as expected. Please check the code for a possible implementaion mistake")
        
        # Initialize consensus parameters and lagrangian multiplies         
        for edge in self._neighbour_edges_id :
            self._consensus_param_neighbours_from_neighbours[edge]   = np.zeros((self._coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ij} in paper)
            self._consensus_param_neighbours_from_self[edge]         = np.zeros((self._coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ji} in paper)
            self._lagrangian_param_neighbours_from_neighbours[edge]  = np.zeros((self._coupling_constraint_size,1)) # lagrangian_coefficient from neighbur 
    
    
    def set_task_constraint_expression(self, constraint : ca.MX) -> None :
        """Save the constraint associated with this parametric task for later retrieval of the lagrangian multipliers (lambda_ji)"""
        
        if constraint.size1() != self._coupling_constraint_size :
            raise ValueError("The constraint size does not match the size of the coupling constraints for the task!")
        self._task_constraint   :ca.MX         = constraint
        
    def step_consensus_variable_from_self(self,neighbour_edge_id:int,learning_rate:float) -> None :
        """Take step of the concensus variable eq3 in Constraint-Coupled Decomposition"""
        self._consensus_param_neighbours_from_self[neighbour_edge_id] -= (self._lagrangian_param_neighbours_from_self - self._lagrangian_param_neighbours_from_neighbours[neighbour_edge_id])*learning_rate
         
    # updating consensus variables
    def update_consensus_variable_from_neighbours(self,neighbour_edge_id:int,consensus_variable:np.ndarray) -> None :
         self._consensus_param_neighbours_from_neighbours[neighbour_edge_id] = consensus_variable
         
    def update_lagrangian_variable_from_neighbours(self,neighbour_edge_id:int,lagrangian_variable:np.ndarray) -> None :
         self._lagrangian_param_neighbours_from_neighbours[neighbour_edge_id] = lagrangian_variable
    
    def update_consensus_variable_from_self(self,neighbour_edge_id:int,consensus_variable:np.ndarray) -> None :
         self._consensus_param_neighbours_from_neighbours[neighbour_edge_id] = consensus_variable  
    
    def update_lagrangian_variable_from_self(self,lagrangian_variable:np.ndarray) -> None :
          self._lagrangian_param_neighbours_from_self= lagrangian_variable
          
        

def get_inclusion_contraint(zeta:ca.MX, task_container:TaskOptiContainer,source:int, target:int) -> ca.MX:
    """ 
    For a given variable zeta it computes the inclusions constraints required to include the zeta variable in the super-level set of the parameteric task
    
    Args:
        zeta (ca.MX): the variable to be included in the super-level set of the task
        task_container (TaskOptiContainer): the task container for the task to be included
        source (int): the source agent id
        target (int): the target agent id
    
    Returns :
        constraints : the inclusion constraint
        
    Note :
        The inclusion constraint depends upon the direction in which the prediacte is considered. this is why you need as an  input the source and target agents
    """

    
    # Only collaborative parametric task check.
    if not isinstance(task_container.task.predicate,CollaborativePredicate) :
        raise ValueError("Only parametric collaborative tasks are allowed to be decomposed. No current support for Individual tasks")
    
    # A @ (x_i-x_j - c) <= b   =>  A @ (e_ij - c) <= b 
    # becomes 
    # A @ (x_j-x_i + c) >= -b   =>  -A @ (e_ji + c) <= b  =>  A_bar @ (e_ji - c_bar) <= b   (same sign of b,A change the sign)
    
    if not task_container.task.is_parametric :
        eta = np.vstack((task_container.task.predicate.center,np.array([1])))
    else :
        eta = task_container.eta_var
    
    # Check the target source pairs as the inclusion relation needs to respect the direction of the task.
    if (source == task_container.task.predicate.source_agent) and (target == task_container.task.predicate.target_agent) :
        A = task_container.task.predicate.A
        b = task_container.task.predicate.b[:,np.newaxis] # makes it a column
        A_z = np.hstack((A,b)) 
        
        constraints = A@zeta - A_z@eta <= ca.DM.zeros((A.shape[0],1)) 
        
    else :
        A = -task_container.task.predicate.A   
        b =  task_container.task.predicate.b[:,np.newaxis] # makes it a column
        A_z = np.hstack((A,b)) 
        
        # The flip matrix reverts the sign of the center variable and lets the scale factor with the same sign.
        flip_mat = ca.diag(ca.vertcat(-ca.DM.ones(task_container.task.state_space_dimension),1))
        
        constraints = A@zeta - A_z@flip_mat@eta <= ca.DM.zeros((A.shape[0],1)) 

    return constraints


def any_parametric( iterable:Iterable[TaskOptiContainer] ) -> bool :
    
    return any([task_container.task.is_parametric for task_container in iterable])

    
T = TypeVar("T", StlTask,TaskOptiContainer )
def get_intersecting_tasks_for(target_task: T, tasks:Iterable[T], let_self: bool=False) -> list[T]:
    """
    Returns the list of tasks that intersect the given task. The intersection is computed by checking the intersection of the time intervals of the tasks.
    
    Args:
        task : StlTask : the task for which we want to find the intersecting tasks
        tasks : Iterable[StlTask] : the list of tasks to check for intersection
        let_self bool : if True then the self task will be added to the list of intersecting tasks if this is present among the list of tasks
    
    Returns:
        intersecting_tasks : list[StlTask] : the list of intersecting tasks
    """
    
    
    intersecting_tasks = []
    
    if not isinstance(target_task,TaskOptiContainer)  :
        for other_task in tasks :
            if isinstance(other_task,TaskOptiContainer) :
                raise ValueError("The target_task is not a container but the list contains containers. This is not allowed")
            
            if not (target_task.temporal_operator.time_interval / other_task.temporal_operator.time_interval).is_empty() :
                if let_self or (target_task != other_task) :
                    intersecting_tasks.append(other_task)
                
    else :
        for other_task in tasks :
            if isinstance(other_task,StlTask) :
                raise ValueError(f"The target_task is of type {TaskOptiContainer.__name__} but the list contains elemtns of type {StlTask.__name__}. This is not allowed")
            
            if not (target_task.task.temporal_operator.time_interval / other_task.task.temporal_operator.time_interval).is_empty() :
                if let_self or (target_task != other_task) :
                    intersecting_tasks.append(other_task)
        
    return intersecting_tasks


T = TypeVar("T", StlTask,TaskOptiContainer )
def is_union_covering(target_task:T, tasks:Iterable[T]) -> bool :
    """
    Check if the task is covered by the union of the tasks in the list.
    
    Args:
        task : StlTask : the task to check for covering
        tasks : Iterable[StlTask] : the list of tasks to check for covering
    
    Returns:
        is_covered : bool : True if the task is covered by the union of the tasks in the list
    """
    
    union         = TimeInterval(a=None,b=None)
        
    if not isinstance(target_task,TaskOptiContainer) :
    
        for other_task in tasks :
            if isinstance(other_task,TaskOptiContainer) :
                raise ValueError("The target_task is not a container but the list contains containers. This is not allowed")
            
            if union.can_be_merged_with(other_task.temporal_operator.time_interval) :
                union = union.union(other_task.temporal_operator.time_interval)
            else: # the union is discontinuous so it cannot be containing
                return False
    
    else :
        for other_task in tasks :
            if isinstance(other_task,StlTask) :
                raise ValueError(f"The target_task is of type {TaskOptiContainer.__name__} but the list contains elemtns of type {StlTask.__name__}. This is not allowed")
            
            if union.can_be_merged_with(other_task.task.temporal_operator.time_interval) :
                union = union.union(other_task.task.temporal_operator.time_interval)
            else: # the union is discontinuous so it cannot be containing
                return False
            
    
    if isinstance(target_task,TaskOptiContainer) :
        if target_task.task.temporal_operator.time_interval in union :
            return True
        else :
            return False
    else :
        if target_task.temporal_operator.time_interval in union :
            return True
        else :
            return False


def get_maximal_sets_of_intersecting_always_tasks(task_containers:list[TaskOptiContainer]) -> list[set[TaskOptiContainer]]:
    
    always_tasks : list[TaskOptiContainer] = [task_container for task_container in task_containers if isinstance(task_container.task.temporal_operator,G)]
    maximal_sets = set()
    
    power_set: list[set[TaskOptiContainer]] = powerset(always_tasks)
    intersecting_sets = list()
    
    for set_i in power_set:
        
        if len(set_i) == 1 : # single tasks are not considered to be intersecting with themself
            continue
        
        intersection = TimeInterval(a = float("-inf"),b = float("inf"))
        for task_container in set_i:
            intersection = intersection / task_container.task.temporal_operator.time_interval
        
        if not intersection.is_empty() :
            intersecting_sets.append(set_i)
    
    maximal_sets = []
    
    for set_i in intersecting_sets :
        is_subset = False
        for set_j in intersecting_sets :
            if set_i.issubset(set_j) and set_i != set_j:
                is_subset = True
                break
        if not is_subset :
            maximal_sets.append(set_i)
    
    return maximal_sets


def get_minimal_sets_of_always_tasks_covering_by_union(task_containers:list[TaskOptiContainer]) -> list[set[TaskOptiContainer]]:
    
    always_tasks_containers    : list[TaskOptiContainer]   = [task_container for task_container in task_containers if isinstance(task_container.task.temporal_operator,G)]
    eventually_task_containers : list[TaskOptiContainer]   = [task_container for task_container in task_containers if isinstance(task_container.task.temporal_operator,F)]
    conflict_sets = list()
    
    
    for eventually_task_container in eventually_task_containers :
        intersecting_always_tasks : list[TaskOptiContainer]      = get_intersecting_tasks_for(eventually_task_container, always_tasks_containers )
        if len(intersecting_always_tasks) == 0 :
            continue    
        power_set_of_intersecting_always_tasks : list[set[TaskOptiContainer]] = powerset(intersecting_always_tasks)
        
        for always_task_containers_set in power_set_of_intersecting_always_tasks :
            if is_union_covering(eventually_task_container , always_task_containers_set) :
                candidate_conflict = always_task_containers_set | {eventually_task_container} #set union
                conflict_sets.append(candidate_conflict)
            
    
    # now take the minimal ones
    minimal_conflict_sets = []
    
    for conflict_set in conflict_sets :
        has_subset = False
        
        for other_set in conflict_sets :
            if other_set.issubset(conflict_set) and conflict_set != other_set :
                has_subset = True
                break
        if not has_subset :
            minimal_conflict_sets.append(conflict_set)
            

    return minimal_conflict_sets
    

def get_maximal_sets_of_always_tasks_covering_by_intersection(task_containers:list[TaskOptiContainer]) -> list[set[TaskOptiContainer]]:
    
    always_tasks_containers    : list[TaskOptiContainer] = [task_container for task_container in task_containers if isinstance(task_container.task.temporal_operator,G)]
    eventually_task_containers : list[TaskOptiContainer] = [task_container for task_container in task_containers if isinstance(task_container.task.temporal_operator,F)]
    conflict_sets = list()
    
    
    for eventually_task_container in eventually_task_containers :
        
        intersecting_always_tasks : list[TaskOptiContainer]      = get_intersecting_tasks_for(eventually_task_container, always_tasks_containers )
        if len(intersecting_always_tasks) == 0 :
            continue    
        power_set_of_intersecting_always_tasks : list[set[TaskOptiContainer]] = powerset(intersecting_always_tasks)
    
        for always_task_containers_set in power_set_of_intersecting_always_tasks :
            
            intersection = TimeInterval(a = float("-inf"),b = float("inf"))
            for task_container in always_task_containers_set :
                intersection = intersection / task_container.task.temporal_operator.time_interval
            
            if  eventually_task_container.task.temporal_operator.time_interval in intersection :
                
                candidate_conflict = always_task_containers_set | {eventually_task_container} #set union
                conflict_sets.append(candidate_conflict)
    
    # now take the minimal ones
    maximal_conflict_sets = []
    
    for conflict_set in conflict_sets :
        is_subset = False
        
        for other_set in conflict_sets :
            if conflict_set.issubset(other_set) and conflict_set != other_set :
                is_subset = True
                break
        if not is_subset :
            maximal_conflict_sets.append(conflict_set)
            

    return maximal_conflict_sets




@dataclass(frozen=True)
class DecompositionParameters :
    """
    Storage class for the parameters of the task decomposition.
    Each agent during the decomposition adopts the learning rate equation : LEARNING_RATE_0  *(1/it)^DECAY_RATE
    
    Args :
        learning_rate_0 : float     initial value of the learning rate
        decay_rate    : float       decay rate of the learning rate (<1)
        penalty_coefficient : float penalty coefficient for the constraints
        use_non_linear_cost : bool  not recommended: used to use nonlinear instead of linear cost function over the scale factor
        communication_radius:float  applied to constraint the parametric tasks
        number_of_optimization_iterations : int number of iterations for the optimization
    """
    
    learning_rate_0 : float     = 0.3  # higher value : increases learning speed bu gives rise to jittering
    decay_rate    : float       = 0.7  # higher value : learning rate decays faster
    penalty_coefficient : float = 100  # higher value : can speed up convergence but too high values will enhance jittering
    use_non_linear_cost : bool  = False # not recommended: used to use nonlinear instead of linear cost function over the scale factor
    communication_radius:float  = 1E5   # practically infinity
    number_of_optimization_iterations : int = 1000
    
    def __post_init__(self) :
        
        if self.learning_rate_0 <= 0 :
            raise ValueError("The learning rate must be positive")
        if self.decay_rate <= 0 :
            raise ValueError("The decay rate must be postive")
        if self.penalty_coefficient <= 0 :
            raise ValueError("The penalty coefficient must be positive")
        if self.communication_radius <= 0 :
            raise ValueError("The communication radius must be positive")
        



class EdgeComputingAgent(Publisher) :
    """
       Agents emplyed for the task decomposition (practically representing two agents sharing the computation of a task)
    """
    def __init__(self,edge_id : int,logger_level: int = logging.INFO, parameters: DecompositionParameters =  DecompositionParameters() ) -> None:
        """
        Args:
            agent_id     : Id of the agent
            logger_level : logging level
            parameters   : DecompositionParameters (see also :class: 'DecompositionParameters')
        """        
        
        super().__init__()
        
        self._optimizer         : ca.Opti          = ca.Opti()    # optimizer for the single agent.
        self._edge_id           : int              = edge_id      # only save the edge as a unique integer.  ex: (4,1) -> 41
        self._task_containers   : list[TaskOptiContainer] = []    # list of task containers.
        self._warm_start_solution : ca.OptiSol   = None
        
        self._parametric_task_count   = 0
        self._communication_radius    = parameters.communication_radius  # practically infinity
        self._use_non_linear_cost     = parameters.use_non_linear_cost
        self._jit                     = False
        self._num_iterations          = parameters.number_of_optimization_iterations
        self._current_iteration       = 0
        self._learning_rate_0         = parameters.learning_rate_0        # initial value of the learning rate 
        self._decay_rate              = parameters.decay_rate         # decay rate of the learning rate (<1)
        self._penalty_coefficient     = parameters.penalty_coefficient # penalty coefficient for the constraints
        
        self._penalty = self._optimizer.variable(1)
        self._optimizer.subject_to(self._penalty>=0)
        self._optimizer.set_initial(self._penalty,40)
        
        self._state_space_dim       = None
        self._penalty_values        = []
        self._cost_values           = []
        self._scale_factors_values  = []
        self._is_initialized_for_optimization = False
        self._zeta_variables = []
        
        self.add_topic("new_consensus_variable")
        self.add_topic("new_lagrangian_variable")
        self._logger = logging.getLogger("Agent-" + str(self.edge_id))
    
    @property
    def edge_id(self):
        return self._edge_id
    
    @property
    def is_initialized_for_optimization(self):
        return self._is_initialized_for_optimization
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def task_containers(self):
        return self._task_containers
    
    @property
    def is_computing_edge(self):
        return bool(self._parametric_task_count)
    
    @property
    def communication_radius(self):
        return self._communication_radius
    

    def add_task_containers(self,task_containers: Iterable[TaskOptiContainer] | TaskOptiContainer) :
        """
        Add task containers to the agent
        """
        if isinstance(task_containers,TaskOptiContainer) :
            self._add_task_container(task_containers)
        elif isinstance(task_containers,Iterable):
            for task_container in task_containers :
                self._add_task_container(task_container)
        else :
            raise ValueError("The task containers must be an iterable of TaskOptiContainer. Given: " + str(type(task_containers)))
 
    def _add_task_container(self,task_container:TaskOptiContainer) :
        
        # Checks.
        if not isinstance(task_container.task.predicate,CollaborativePredicate) :
            message = "Only collaborative tasks are allowed as these ar the only ones that should be decomposed."
            self._logger.error(message)
            raise RuntimeError(message)
        
        if task_container.task.is_parametric and (not task_container.has_parent_task) :
            message = "The task container provided contains a parametric task but does not have an assigned parent task. You must set the parent task together with the decomposition path using the method set_parent_task_and_decomposition_path"
            self._logger.error(message)
            raise RuntimeError(message)
        
        #count parametric tasks
        if task_container.task.is_parametric :
            self._parametric_task_count += 1
        
        task_edge_id = edge_to_int((task_container.task.predicate.source_agent,task_container.task.predicate.target_agent))
        if task_edge_id != self._edge_id :
            message = f"The task container with collaborative task over the edge {(task_edge_id)} does not belong to this the edge"
            self._logger.error(message)
            raise RuntimeError(message)
        
        if len(self._task_containers) == 0:
            self._state_space_dim = task_container.task.predicate.state_space_dim #! todo: we will need to add a check that all the tasks have the state state space 
            
        self._task_containers.append(task_container) 
            
    def _compute_shared_constraints(self) -> list[ca.MX]:
        """
        Computes the shared inclusion constraint for the given agent. The shared constraints are the inclusion of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints : set of constraints due to the parametric task path constraints
        """        

        constraints_list = []
        for container in self._task_containers :
            if container.task.is_parametric :
               
                task                 = container.task                          # extract task
                num_computing_agents = container.length_decomposition_path -1  # Number of agents sharing the computation for this constraint
                # Compute constraints.
                M,Z  = get_M_and_Z_matrices_from_inclusion(P_including=container.parent_task, P_included=task) # get the matrices for the inclusion constraint
                Z    = Z/num_computing_agents # scale the Z matrix by the number of computing agents 
       
                penalty_vec  = ca.DM.ones((container.average_consensus_param.size1(),1)) * self._penalty 
                zero_vec     = ca.DM.zeros((container.average_consensus_param.size1(),1))
            
                constraint   = (M@container.eta_var - Z)  - penalty_vec  + container.average_consensus_param <= zero_vec  # set the constraint
                constraints_list += [constraint]
                container.set_task_constraint_expression(constraint=constraint)
        
        return constraints_list
        
    def _compute_communication_consistentcy_constraints(self) -> list[ca.MX]:
        
        constraints = []
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                task = task_container.task
                Nk_list = communication_consistency_matrices_for(task = task)
                
                for Nk in Nk_list :
                    constraints += [  task_container.eta_var.T@Nk@task_container.eta_var <= self._communication_radius**2 ]
        
        return constraints
            
    def _compute_overloading_constraints(self) -> list[ca.MX]:
        """
        Returns constraints for overloading of the edge with multiple collaborative tasks

        Returns:
            constraints : overloading constraints
        """        
        
        constraints = []
        conflict_sets_L = get_maximal_sets_of_intersecting_always_tasks(task_containers = self._task_containers)
        conflict_sets_C = get_minimal_sets_of_always_tasks_covering_by_union(   task_containers = self._task_containers)
        conflict_sets_D = get_maximal_sets_of_always_tasks_covering_by_intersection(task_containers = self._task_containers)
        
        # print length of the sets 
        conflict_sets = conflict_sets_L + conflict_sets_C + conflict_sets_D
        self._number_of_conflict_sets = len(conflict_sets)
        
        clean_conflict_sets = list()
        for conflict_set in conflict_sets :
            if not (conflict_set in clean_conflict_sets) :
                clean_conflict_sets.append(conflict_set)
                
        self._number_of_conflict_sets = len(clean_conflict_sets)
        
        
        
        for conflict_set in  clean_conflict_sets :
            
            if any_parametric(conflict_set) :
                is_first = True
                zeta = self._optimizer.variable(self._state_space_dim)* self.communication_radius #todo! change the number 2 with state space dimension when moving to higher dimensional systems
                self._zeta_variables += [zeta]
                
                for task_container in conflict_set :
                    if is_first : # they can be aligned with the first task in the conflict set
                        i,j = task_container.task.predicate.source_agent,task_container.task.predicate.target_agent
                        is_first = False
                    
                    constraints += [get_inclusion_contraint(zeta = zeta, task_container = task_container,source=  i,target=j)]
                        
        return constraints   
    
    
    
            
  
    
    def setup_optimizer(self, jit: bool = False) -> None :  
        """
        setup the optimization problem for the given agent
        """    
        
        # If there is not prametric task you can skip
        if not self.is_computing_edge :
            return  
        
        self._jit = jit
        
        if self._is_initialized_for_optimization :
            message = "The optimization problem was already set up. You can only set up the optimization problem once at the moment."
            self._logger.error(message)
            raise RuntimeError(message)
        
        if len(self._task_containers) == 0 :
            message = "The agent does not have any tasks to decompose. The optimization problem will not be setup and solved."
            self._logger.warning(message)

        
        self._is_initialized_for_optimization  = True
        
        if self._use_non_linear_cost:
            cost = 1
        else :
            cost = 0
        
        for task_container in self._task_containers :
            
            if task_container.task.is_parametric:
                task_container.set_optimization_variables(self._optimizer) # creates scale,center and consensus variables for each task.
        
                # Set the scale factor positive in constraint
                epsilon = 1e-3
                self._optimizer.subject_to(task_container.scale_var >= epsilon)
                self._optimizer.subject_to(task_container.scale_var <= 1)          # this constraint is not needed in theory as the optimal solution must abide bide this constraint. But it helps convergence.
                
                # set an initial value for the scale variables
                scale_guess =1
                center_guess = task_container.parent_task.predicate.center/len(task_container.neighbour_edges_id)
                
                self._optimizer.set_initial(task_container.scale_var , scale_guess) # set an initial guess value
                self._optimizer.set_initial(task_container.center_var, center_guess) # set an initial guess value for the center variable
                
                if self._use_non_linear_cost:
                    cost *= 1/task_container.scale_var**2
                else :
                    cost += -10*task_container.scale_var 
            
        cost += self._penalty_coefficient*self._penalty # setting cost toward zero
        self._optimizer.minimize(cost)
        
        # set up private and shared constraints (also if you don't have constraints this part will be printed)
        overloading_constraints   = self._compute_overloading_constraints()
        shared_path_constraint    = self._compute_shared_constraints()
        communication_constraints = self._compute_communication_consistentcy_constraints()
        
        if len(overloading_constraints) != 0:
            for constraint in overloading_constraints :
                self._optimizer.subject_to(constraint)
        
        # safety check 
        if len(shared_path_constraint) == 0 or len(communication_constraints) == 0 :
            raise RuntimeError("An edge marked as computing does not have shared constraints or communication consistency constraints. Contact the developers for a possible implementation error.")
        
        for constraint in shared_path_constraint :
            self._optimizer.subject_to(constraint)
        for constraint in communication_constraints :
            self._optimizer.subject_to(constraint)
        

        print("-----------------------------------")
        print(f"Computing edge                          : {self._edge_id}")
        print(f"Number of overloading_constraints       : {len(overloading_constraints)}")
        print(f"Number of shared constraints            : {len(shared_path_constraint )}")
        print(f"Number of variables                     : {self.optimizer.nx}")
        print(f"Number of parameters  (from concensus)  : {self.optimizer.np}")
        print(f"Number of parametric tasks              : {self._parametric_task_count}")
        print(f"Number of conflicting conjunctions sets : {self._number_of_conflict_sets}")
        
        if self._jit :
            # Problem options
            p_opts = dict(print_time=False, 
                        verbose=False,
                        expand=True,
                        compiler='shell',
                        jit=True,  # Enable JIT compilation
                        jit_options={"flags": '-O3', "verbose": True, "compiler": 'gcc'})  # Specify the compiler (optional, default is gcc))

        else :
            # Problem options
            p_opts = dict(print_time=False, 
                        verbose=False,
                        expand=True)
        
        
        # Solver options
        s_opts = dict(
            print_level=1,
            tol=1e-6,
            max_iter=1000,
            )

        self._optimizer.solver("ipopt",p_opts,s_opts)
        self._cost = cost
        
        
    def solve_local_problem(self,print_result:bool=False) -> None :
        
        # skip computations and add to counter
        if not self.is_computing_edge :
            self._current_iteration += 1
            return  

            
        # Update the consensus parameters.
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                consensus_average = task_container.compute_consensus_average_parameter() 
                self._optimizer.set_value(task_container.average_consensus_param,   consensus_average )
                
        if self._warm_start_solution != None :
            try :
                self._optimizer.set_initial(self._warm_start_solution.value_variables())
            except Exception as e:
                message = "An error occured while setting the warm start solution. The error is probably due to the fact that the warm start solution has some nana or inf values."
                solution_print = f"The warm start solution is: {self._warm_start_solution.value_variables()}"
                self._logger.error(message + "\n" + solution_print)
                raise RuntimeError(message + "\n" + solution_print)
         

        try :
            with contextlib.redirect_stdout(f):
                sol : ca.OptiSol = self._optimizer.solve() 
                # Reset the StringIO buffer
                f.seek(0)
                f.truncate(0)
                
        except  Exception as e:
            self._logger.error(f"An error occured while solving the optimization problem for agent {self._edge_id}. The error message is: {e}")
            raise e
        
        if print_result :
            print("--------------------------------------------------------")
            print(f"Agent {self._edge_id } SOLUTION")
            print("--------------------------------------------------------")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    print("Center and Scale")
                    print(self._optimizer.value(task_container.center_var))
                    print(self._optimizer.value(task_container.scale_var))
                    for zeta in self._zeta_variables :
                        print("Zeta")
                        print(self._optimizer.value(zeta) * self.communication_radius)
            print("penalty")
            print(self._optimizer.value(self._penalty))
            print("--------------------------------------------------------")
        
        
        parametric_tasks = [task_container for task_container in self._task_containers if task_container.task.is_parametric]
        currents_scale = {}
        
        for parametric_tasks in parametric_tasks :
            currents_scale[parametric_tasks.parent_task_id] = self._optimizer.value(parametric_tasks.scale_var)
            
        self._scale_factors_values += [currents_scale]
        self._penalty_values       += [self._optimizer.value(self._penalty)]
        self._cost_values          += [self._optimizer.value(self._cost)]
        
        
        self._warm_start_solution = sol
        
        # Update lagrangian coefficients
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                
                lagrangian_coefficient = self._optimizer.value(self._optimizer.dual(task_container.local_constraint))[:,np.newaxis]
                task_container.update_lagrangian_variable_from_self(lagrangian_variable=lagrangian_coefficient) # Save the lagrangian variable associated with this task.
        
        
        # Update current iteration count.
        self._current_iteration += 1
    
    
    def step_consensus_variables_with_currently_available_lagrangian_coefficients(self) -> None :
        learning_rate = self._learning_rate_0 *(1/self._current_iteration**self._decay_rate)
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                for edge_id in task_container.neighbour_edges_id :
                    task_container.step_consensus_variable_from_self(neighbour_edge_id = edge_id,learning_rate = learning_rate)
                    
    
    # Callbacks and notifications.
    def update_consensus_variable_from_edge_neighbour_callback(self,consensus_variables_map_from_neighbour:dict[int,np.ndarray],neighbour_edge_id:int, parent_task_id:int) -> None :
        """Update the task of the agent"""
        self._logger.debug(f"Receiving consensus variable from neighbour edge {neighbour_edge_id} for parent task {parent_task_id}")
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                if task_container.parent_task_id == parent_task_id : # you have a task connected to the same parent id.
                    # From the variables of the neighbur extract the one correspoding to your edge.
                    consensus_variable = consensus_variables_map_from_neighbour[self._edge_id]
                    # Update the consensus variables as received from the neighbour.
                    task_container.update_consensus_variable_from_neighbours(neighbour_edge_id=  neighbour_edge_id , consensus_variable= consensus_variable)
                    break
        
    def update_lagrangian_variable_from_edge_neighbour_callback(self,lagrangian_variable_from_neighbour:np.ndarray,neighbour_edge_id:int,parent_task_id:int) -> None :
        """Update the task of the agent"""
        self._logger.debug(f"Receiving lagrangian variable from neighbour edge {neighbour_edge_id} for parent task {parent_task_id}")
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                if task_container.parent_task_id == parent_task_id :
                    task_container.update_lagrangian_variable_from_neighbours(neighbour_edge_id = neighbour_edge_id,lagrangian_variable= lagrangian_variable_from_neighbour)
                    break
        
    
    def notify(self,event_type:str):
        if event_type == "new_consensus_variable" :
            self._logger.debug(f"Notifying neighbours about the new consensus variable")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    for callback in self._subscribers[event_type] :
                        callback( task_container._consensus_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
                    
                    
        elif event_type == "new_lagrangian_variable" :
            self._logger.debug(f"Notifying neighbours about the new lagrangian variable")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    
                    for callback in self._subscribers[event_type] :
                        callback( task_container._lagrangian_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
                        
                        
                        
    def deparametrize_container(self,task_container : TaskOptiContainer) -> StlTask | None :
        """ Once the decomposition occurred just task the value of the optimal parametrs and deparametrize the tasks that have to be parameterized"""
        
        if task_container.task.is_parametric :
            try :
                center = np.asanyarray(self._optimizer.value(task_container.center_var))
                scale  = float(self._optimizer.value(task_container.scale_var))
            except Exception as e :
                message = f"Error trying to retrieve the optimization variables for task {task_container.task.task_id}. Probably the optimization was not run yet. Error: {e}"
                self._logger.error(message)
                raise RuntimeError(message)
            
            A = task_container.task.predicate.A
            b = task_container.task.predicate.b*scale
            polytope_0    =  pc.Polytope(A = A, b = b)
            new_predicate = CollaborativePredicate(polytope_0=polytope_0 , 
                                                   center = center,
                                                   source_agent_id = task_container.task.predicate.source_agent, 
                                                   target_agent_id = task_container.task.predicate.target_agent)
            new_task = StlTask(predicate = new_predicate, temporal_operator = task_container.task.temporal_operator)
        else :
            # if task is not parametric return the task as it is
            new_task = task_container.task
        return new_task
        
    def get_parametric_container_with_parent_task(self,parent_task_id:int) -> TaskOptiContainer | None :
        
        for task_container in self._task_containers :
            if task_container.task.is_parametric:
                if task_container.parent_task_id == parent_task_id :
                    return task_container
        return None
    
    
    
    
class EdgeComputingGraph(nx.Graph) :
    """Just a support class to easily manage the computing graph"""
    def __init__(self,incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
    
    def add_node(self, node_for_adding, parameters: DecompositionParameters, **attr):
        """ Adds an edge to the graph."""
        super().add_node(node_for_adding, **attr)
        self._node[node_for_adding][AGENT] = EdgeComputingAgent(edge_id = node_for_adding,parameters = parameters)
        
        
def extract_computing_graph_from_communication_graph(communication_graph :CommunicationGraph, parameters: DecompositionParameters) -> EdgeComputingGraph :
        
    if not nx.is_tree(communication_graph) :
        raise ValueError("The communication graph must be acyclic to obtain a valid computation graph")
    
    computing_graph = EdgeComputingGraph()
    selfloops = set(nx.selfloop_edges(communication_graph,data=False))
    
    edge_view = communication_graph.edges
    
    edges     = set(communication_graph.edges)
    edges     = edges - selfloops
    
    
    for edge in edges :
        computing_graph.add_node(edge_to_int(edge),parameters=parameters)
    
    for edge in edges :    
        # Get all edges connected to node1 and node2
        edges_node1 = set(edge_view(edge[0]))
        edges_node2 = set(edge_view(edge[1]))
    
        # Combine the edges and remove the original edge
        adjacent_edges = list((edges_node1 | edges_node2) - {edge, (edge[1],edge[0])})
        computing_edges = [ (edge_to_int(edge), edge_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    return computing_graph
    
    
    
    

def run_task_decomposition(communication_graph:nx.Graph,task_graph:nx.Graph,parameters : DecompositionParameters = DecompositionParameters(),logger_file:str = None,logger_level:int = logging.INFO, jit:bool= False) :
    """Task decomposition pipeline"""
    
    # Normalize the task graph and the communication graph to have the same nodes as a precaution.
    normalize_graphs(communication_graph, task_graph)
    
    # remove the self edges from the communicaiton graph to check acyclicity
    selfloops = nx.selfloop_edges(communication_graph)
    communication_graph.remove_edges_from(selfloops)
    
    # Check the communication graph.
    if not nx.is_connected(communication_graph) :
        raise ValueError("The communication graph is not connected. Please provide a connected communication graph")
    if not nx.is_tree(communication_graph) :
        raise ValueError("The communication graph is not a acyclic. Please provide an acyclic communication graph")
    
    communication_graph.add_edges_from(selfloops)
    original_task_graph :TaskGraph = clean_task_graph(task_graph) # remove edges that do not have tasks to keep the graph clean..
    new_task_graph      :TaskGraph = TaskGraph()

    # Create computing graph. Communication radius important to ensure communication consistency of the solution (and avoid diverging iterates).
    computing_graph :EdgeComputingGraph = extract_computing_graph_from_communication_graph(communication_graph = communication_graph, parameters = parameters)

    critical_task_to_path_mapping : dict[StlTask,list[int]] = {} # mapping of critical tasks to the path used to decompose them.
    count_new_tasks_added = 0
    # For each each edge check if there is a decomposition to be done.
    for edge in original_task_graph.edges : 
        
        
        if edge[0] == edge[1] : # if the edge is a self edge.
            task_list = original_task_graph[edge[0]][edge[1]][MANAGER].tasks_list
            new_task_graph.add_edge(edge[0],edge[1])
            # add the tasks to the new task graph.
            new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task_list)
        
        elif (edge in communication_graph.edges): # if the edge is consistent with the communication graph.
            task_list = original_task_graph[edge[0]][edge[1]][MANAGER].tasks_list
            new_task_graph.add_edge(edge[0],edge[1])
            # add the tasks to the new task graph.
            new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task_list)
            
            # Add tasks to the computing graph to go on with the decomposition (if not self tasks).
            if edge[0] != edge[1] :
                for task in task_list :
                    container = TaskOptiContainer(task = task) # non parametric task containers.
                    computing_graph.nodes[edge_to_int((edge[1],edge[0]))][AGENT].add_task_containers(task_containers = container)
        
        else :  # if the edge is a critical edge.
            # Retrieve all the tasks on the edge because such tasks will be decomposed.
            tasks_to_be_decomposed: list[StlTask] =  original_task_graph[edge[0]][edge[1]][MANAGER].tasks_list
            
            path = nx.shortest_path(communication_graph,source = edge[0],target = edge[1]) # find shortest path connecting the two nodes.
            edges_through_path = edge_set_from_path(path=path) # find edges along the path.
            
            for parent_task in tasks_to_be_decomposed : # add a new set of tasks for each edge along the path of the task to be decomposed.
                
                critical_task_to_path_mapping[parent_task] = path # map the critical task to the path used to decompose it.
                if (parent_task.predicate.source_agent,parent_task.predicate.target_agent) != (path[0],path[-1]) : # case the direction is incorrect.
                    parent_task.flip() # flip the task such that all tasks have the same direction of the node.
                
                for source_node,target_node in  edges_through_path :
                    new_task_graph.add_edge(source_node,target_node) # adds the edge (nothing doe in case the edge is already present)
                    
                    # Create parametric task along this edge and add it to the communication graph.
                    subtask = create_parametric_collaborative_task_from(task = parent_task , source_agent_id = source_node, target_agent_id = target_node ) 
                    count_new_tasks_added +=1
                    
                    # Create task container to be given to the computing agent corresponding to the edge.
                    task_container = TaskOptiContainer(task=subtask)
                    task_container.set_parent_task_and_decomposition_path(parent_task = parent_task,decomposition_path = path)
                    computing_graph.nodes[edge_to_int((source_node,target_node))][AGENT].add_task_containers(task_containers = task_container)
        
    
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for node in computing_graph.nodes :
        computing_graph.nodes[node][AGENT].setup_optimizer(jit = jit)
    
    
    # Set the callback connections.
    computing_agents_couples = list(combinations(computing_graph.nodes,2))
    for edge_i,edge_j in computing_agents_couples:
        
        if edge_j in computing_graph.neighbors(edge_i) :
            
            agent_i = computing_graph.nodes[edge_i][AGENT]
            agent_j = computing_graph.nodes[edge_j][AGENT]
            
            agent_i.register("new_consensus_variable",agent_j.update_consensus_variable_from_edge_neighbour_callback)
            agent_i.register("new_lagrangian_variable",agent_j.update_lagrangian_variable_from_edge_neighbour_callback)
            
            agent_j.register("new_consensus_variable",agent_i.update_consensus_variable_from_edge_neighbour_callback)
            agent_j.register("new_lagrangian_variable",agent_i.update_lagrangian_variable_from_edge_neighbour_callback)



    # Solution loop.
    for jj in tqdm(range(parameters.number_of_optimization_iterations )) :
        
        # Everyone shares the current consensus variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].notify("new_consensus_variable")
            
        # Everyone solves the local problem and updates the lagrangian variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].solve_local_problem(  print_result=False )
            
        # Everyone updates the value of the consensus variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].notify("new_lagrangian_variable")
        
        # Everyone updates the consensus variables using the lagrangian coefficients updates.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].step_consensus_variables_with_currently_available_lagrangian_coefficients()
    
    
    
 
    decomposition_accuracy_per_task = {}
    
    for iteration in range(parameters.number_of_optimization_iterations) :
        for edge_id in computing_graph.nodes :
            agent = computing_graph.nodes[edge_id][AGENT]
            scale_factors = agent._scale_factors_values
            for parent_task_id,scale_value in scale_factors[iteration].items() :
                task_accuracy = decomposition_accuracy_per_task.setdefault(parent_task_id,np.zeros((parameters.number_of_optimization_iterations,)))
                task_accuracy[iteration] += scale_value
        
    
    agents_cost_and_penalties = {}
    # Extract solution.   
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    
    for edge_id in computing_graph.nodes :
        results_per_agent = {}
        if not computing_graph.nodes[edge_id][AGENT].is_computing_edge :
            continue
        
        agent = computing_graph.nodes[edge_id][AGENT]
        edge  = (agent.task_containers[0].task.predicate.source_agent,agent.task_containers[0].task.predicate.target_agent)
        
        if agent.is_initialized_for_optimization :
            ax1.plot(range( parameters.number_of_optimization_iterations),agent._penalty_values,label=f"Agent :{edge_id}")
            ax1.set_ylabel("penalties")
            
            ax2.plot(range( parameters.number_of_optimization_iterations),agent._cost_values,label=f"Agent :{edge_id}")
            ax2.set_ylabel("cost")
        
        results_per_agent["penalties"]  = agent._penalty_values
        results_per_agent["cost"]       = agent._cost_values
        results_per_agent["iterations"] = tuple(range(parameters.number_of_optimization_iterations))
    
        agents_cost_and_penalties[edge]                   = results_per_agent
    
    ax1.legend()
    ax2.legend()
    
    for task_id, accuracy in decomposition_accuracy_per_task.items() :
            
        ax3.plot(range( parameters.number_of_optimization_iterations),accuracy,label=f"Task :{task_id}")
        ax3.set_ylabel("accuracy")
    
    
    # Fill the new_task graph.
    for edge in new_task_graph.edges :
        if not (edge[0] == edge[1]) :
            computing_edge_id = edge_to_int(edge)
            for container in computing_graph.nodes[computing_edge_id][AGENT].task_containers :
                task = computing_graph.nodes[computing_edge_id][AGENT].deparametrize_container(task_container = container) # Returns the tasks (deparametrised in case it is parametric).
                new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task) # add the task to the new task graph once the solution is found.
    
    new_task_graph = clean_task_graph(new_task_graph) # clean the graph from edges without any tasks (so the ones present in the old task graph and that are not used now)
    
    
    print("**************************************************************")
    print("General Informations")
    print("**************************************************************")
    print("Number of critical tasks : ",len(critical_task_to_path_mapping))
    print("Number of new tasks added :", count_new_tasks_added)
    print("**************************************************************")
    print("Specific information")
    print("**************************************************************")
    
    # print the decomposition result
    for critical_task,path in critical_task_to_path_mapping.items() :
        
        center_sum = np.array([0.,0.])
        scale_sum  = 0
        
        print("**************************************************************")
        print("Critical task :",critical_task.task_id)
        for edge in edge_set_from_path(path) :
            edge_id = edge_to_int(edge)
            agent : EdgeComputingAgent = computing_graph.nodes[edge_id][AGENT]
            task_container    = agent.get_parametric_container_with_parent_task(parent_task_id = critical_task.task_id)
            task = task_container.task
            
            if task.predicate.source_agent != edge[0] :
                center = -np.asarray(agent.optimizer.value(task_container.center_var)).flatten()
                scale  = float(agent.optimizer.value(task_container.scale_var))
            else :
                center = np.asarray(agent.optimizer.value(task_container.center_var)).flatten()
                scale  = float(agent.optimizer.value(task_container.scale_var))
                
            center_sum += center
            scale_sum  += scale
        
        print(f"Number of hyperplanes  : {len(critical_task.predicate.A)}")
        print(f"Temporal operator      : {critical_task.temporal_operator}")
        print(f"Original edge          : {(critical_task.predicate.source_agent,critical_task.predicate.target_agent)}")   
        print(f"Decomposition path     : {path}")
        print(f"Original center        : {critical_task.predicate.center.flatten()}")
        print(f"Center summation       : {center_sum}")  
        print(f"Decomposition accuracy : {scale_sum}")
        print(f"Final Penalty value    : {agent.optimizer.value(agent._penalty)}")
        print("Make sure to check the penalties plot for convergence.\n If the penalties did not go to zero, then the optimization\n did not converge to constraint satisfaction.")
    
    return new_task_graph,computing_graph,  agents_cost_and_penalties,decomposition_accuracy_per_task 


    
if __name__== "__main__" :
    
    
    # Test power set
    print("Testing powerset")
    print(list(powerset({1,2,3,4,5})))
    
    # Test Task container
    print("Testing Task Container")
    i = 3
    j = 2
    A,b = regular_2D_polytope(number_hyperplanes=3,distance_from_center=2)
    t_operator = G(a = 0,b = 10)
    predicate  = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=i,target_agent_id=j)
    
    task      = StlTask(temporal_operator = t_operator,predicate=predicate)
    container = TaskOptiContainer(task = task)
    
    pred2     = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=i,target_agent_id=j)
    task2     = StlTask(temporal_operator = t_operator,predicate=pred2)
    
    opti = ca.Opti()
    print(container.task.is_parametric)
    print(container.has_parent_task)
    container.set_parent_task_and_decomposition_path(parent_task = task2,decomposition_path = [(1,2),(2,3),(3,5),(5,7)])
    print(container.has_parent_task)
    container.set_optimization_variables(opti)
    
    container.center_var
    container.scale_var
    print(container.parent_task_id)
    print(container.task.task_id)
    print(container.neighbour_edges_id)
    container.step_consensus_variable_from_self(12,0.9)
    
    
    
    