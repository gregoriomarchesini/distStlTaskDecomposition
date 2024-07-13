import numpy as np
import casadi as ca
from   itertools import chain,combinations
import networkx as nx 
import logging
from typing import Iterable
import matplotlib.pyplot as plt
from math import floor
from tqdm import tqdm
import contextlib
import io


from stlddec.stl_task import * 
from stlddec.transport import Publisher
from stlddec.graphs import *
from stlddec.utils import *



# Create a reusable StringIO buffer
f = io.StringIO()

def edge_set_from_path(path:list[int],isCycle:bool=False) -> list[(int,int)] :
    
    if not isCycle :
      edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
    elif isCycle : # due to how networkx returns edges
      edges = [(path[i],path[i+1]) for i in range(-1,len(path)-1)]
        
    return edges

def powerset(iterable) -> list[set]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    power_set = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    power_set = [set(x) for x in power_set] # use set operations
    return power_set


class TaskOptiContainer :
    """Storage of variables for each task to be decomposed (Only collaborative tasks)"""
    
    def __init__(self, task : StlTask, center_scale_factor: float = 1.) -> None:
    def __init__(self, task : StlTask, center_scale_factor: float = 1.) -> None:
        
        """
        """    
         
         
        if center_scale_factor <= 0 :
            raise ValueError("The scale factor must be positive")
        
        if not isinstance(task.predicate,CollaborativePredicate):
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
        
         
        if center_scale_factor <= 0 :
            raise ValueError("The scale factor must be positive")
        
        if not isinstance(task.predicate,CollaborativePredicate):
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
        
        self._task = task
        # These attributes are enables only when the task is parametric
        self._parent_task            :"StlTask"            = None  # instance of the parent class
        self._decomposition_path     :list[int]            = []    # path of the decomposition of the task when the task is parametric
        self._neighbour_edges_id     :list[int]            = []    # list of the edges neighbouring to the task edge and expressed as a single id. Ex : (4,1)-> (41) 
        self._center_scale_factor    :float                = center_scale_factor # this is used to scale the center_var variable in order to obtain better convergence (usually the communication radiu is used as scaling factor).
        self._center_scale_factor    :float                = center_scale_factor # this is used to scale the center_var variable in order to obtain better convergence (usually the communication radiu is used as scaling factor).
        
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
    
    
    def set_optimization_variables(self,opti:ca.Opti, scale_factor:float = 1.0) -> None:
    def set_optimization_variables(self,opti:ca.Opti, scale_factor:float = 1.0) -> None:
        """Set the optimization variables for the task container"""
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric hence the optimization variables cannot be set")
        
        if scale_factor <= 0 :
            raise ValueError("The scale factor must be positive")
        
        self._center_scale_factor = scale_factor
        self._center_var   = opti.variable(self.task.predicate.state_space_dim,1)*self._center_scale_factor                      # center the parameteric formula
        self._scale_var    = opti.variable(1)                                                         # scale for the parametric formula
        if scale_factor <= 0 :
            raise ValueError("The scale factor must be positive")
        
        self._center_scale_factor = scale_factor
        self._center_var   = opti.variable(self.task.predicate.state_space_dim,1)*self._center_scale_factor                      # center the parameteric formula
        self._scale_var    = opti.variable(1)                                                         # scale for the parametric formula
        self._eta_var     = ca.vertcat(self._center_var,self._scale_var)                                    # optimization variable
        
        coupling_constraint_size      = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        self._average_consensus_param = opti.parameter(coupling_constraint_size,1) # average consensus parameter for the concesusn variable y
        self._is_initialized          = True

    
    def compute_consensus_average_parameter(self) -> np.ndarray :
        """computes sum_{j} \lambda_{ij} - \lambda_{ji}  
        
        Args: 
            consensus_variable_from_neighbours (EdgeDict) : consensus variable transmitted from the neighbours (\lambda_{ji} in paper)
        Returns:
            average (np.ndarray) : average of the consensus variable
        
        """
        average = 0
        for edge_id in self._neighbour_edges_id:
            average += self._consensus_param_neighbours_from_self[edge_id] - self._consensus_param_neighbours_from_neighbours[edge_id]
            
        return average
    
    def set_parent_task_and_decomposition_path(self,parent_task:"StlTask", decomposition_path:list[tuple[int,int]])-> None:
        
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
        """Save the constraint for later retrieval of the lagrangian multipliers"""
        
        if constraint.size1() != self._coupling_constraint_size :
            raise ValueError("The constraint size does not match the size of the coupling constraints for the task!")
        self._task_constraint   :ca.MX         = constraint
        
    def step_consensus_variable_from_self(self,neighbour_edge_id:int,learning_rate:float) -> None :
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
        flip_mat = ca.diag(ca.vertcat(-ca.DM.ones(task_container.task.state_space_dimension),1))
        
        constraints = A@zeta - A_z@flip_mat@eta <= ca.DM.zeros((A.shape[0],1)) 

    return constraints


def any_parametric( iterable:Iterable[TaskOptiContainer] ) -> bool :
    
    return any([task_container.task.is_parametric for task_container in iterable])

    
    

class EdgeComputingAgent(Publisher) :
    """
       Edge Agent Task Decomposition
    """
    def __init__(self,edge_id : int,  logging_file: str= None,logger_level: int = logging.INFO) -> None:
    def __init__(self,edge_id : int,  logging_file: str= None,logger_level: int = logging.INFO) -> None:
        """
        Args:
            agentID (int): Id of the agent
        """        
        
        super().__init__()
        
        self._optimizer         : ca.Opti          = ca.Opti()    # optimizer for the single agent.
        self._edge_id           : int              = edge_id      # only save the edge as a unique integer.  ex: (4,1) -> 41
        self._task_containers   : list[TaskOptiContainer] = []    # list of task containers.
        self._parametric_task_count = 0
        self._communication_radius  = 100000                      # practically infinity
        self._parametric_task_count = 0
        self._communication_radius  = 100000                      # practically infinity
        self._warm_start_solution : ca.OptiSol   = None
        self._switch_to_SQPMETHOD     = False # in most of the situation does not work well with output error "Search_Direction_Becomes_Too_Small" (but if working is much faster then ipopt).
        self._use_non_linear_cost     = False

        self._num_iterations      = None
        self._current_iteration   = 0
        self._learning_rate_0     = 2           # initial value of the learning rate 
        self._decay_rate          = 0.7         # decay rate of the learning rate (<1)
        self._learning_rate_0     = 10         # initial value of the learning rate    (if the problem does not converge try to increase this value). 
        self._decay_rate          = 0.3         # decay rate of the learning rate (<1)
        
        self._penalty = self._optimizer.variable(1)
        self._optimizer.subject_to(self._penalty>=0)
        self._optimizer.set_initial(self._penalty,40)
        
        self._penalty_values = []
        self._cost_values    = []
        self._is_initialized_for_optimization = False
        self._zeta_variables = []
        
        self.add_topic("new_consensus_variable")
        self.add_topic("new_lagrangian_variable")
        self._logger = get_logger("Agent-" + str(self.edge_id),output_file=logging_file,level = logger_level)
    
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
    
    @communication_radius.setter
    def communication_radius(self,new_radius:float):
        
        if new_radius <= 0 :    
            raise ValueError("The communication radius must be positive")
        self._communication_radius = float(new_radius)
    
    
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
        
        return bool(self._parametric_task_count)
    
    @property
    def communication_radius(self):
        return self._communication_radius
    
    @communication_radius.setter
    def communication_radius(self,new_radius:float):
        
        if new_radius <= 0 :    
            raise ValueError("The communication radius must be positive")
        self._communication_radius = float(new_radius)
    
    
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
        
        #count parametric tasks
        if task_container.task.is_parametric :
            self._parametric_task_count += 1
        
        task_edge_id = edge_to_int((task_container.task.predicate.source_agent,task_container.task.predicate.target_agent))
        if task_edge_id != self._edge_id :
            message = f"The task container with collaborative task over the edge {(task_edge_id)} does not belong to this the edge"
            self._logger.error(message)
            raise RuntimeError(message)
            
        self._task_containers.append(task_container) 
        self._task_containers.append(task_container) 
            
    def _compute_shared_constraints(self) -> list[ca.MX]:
        """computes the shared inclusion constraint for the given agent. The shared constraints are the incluson of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints (list[ca.Function]): set of constraints
        """        
        
        constraints_list = []
        for container in self._task_containers :
            if container.task.is_parametric :
                task                 = container.task                          # extract task
                num_computing_agents = container.length_decomposition_path -1  # Number of agents sharing the computation for this constraint
                print("Number of computing agents: ",num_computing_agents)
                # Compute constraints.
                M,Z  = get_M_and_Z_matrices_from_inclusion(P_including=container.parent_task, P_included=task) # get the matrices for the inclusion constraint
                Z    = Z/num_computing_agents # scale the Z matrix by the number of computing agents 
       
                penalty_vec  = ca.DM.ones((container.average_consensus_param.size1(),1)) * self._penalty 
                zero_vec     = ca.DM.zeros((container.average_consensus_param.size1(),1))
            
                constraint   = (M@container.eta_var - Z)  - penalty_vec  + container.average_consensus_param <= zero_vec  # set the constraint
                constraints_list += [constraint]
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
        """multiple collaborative tasks

        Returns:
            constraints (list[ca.MX]): Returns constraints for overloading of the edge with multiple collaborative tasks
        """        
        
        constraints = []
        
        always_task_containers = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,AlwaysOperator)]        
        
        # Check the always intersections between tasks:
        if len(self._task_containers) == 1 :
           return  [] # with one single task you don't need overloading constraints
        
        
        # 1) Always intersection constraints.
        always_tasks_combinations : list[tuple[TaskOptiContainer,TaskOptiContainer]] = list(combinations(always_task_containers,2))
        
        for container1,container2 in always_tasks_combinations :
            time_intersection = container1.task.temporal_operator.time_interval / container2.task.temporal_operator.time_interval
            if  ( not time_intersection .is_empty()) and (any_parametric([container1,container2])): # there is an intersection
                
                zeta = self._optimizer.variable(container1.task.state_space_dimension) * self.communication_radius # scaled by the communication radius for better convergence. * self.communication_radius # scaled by the communication radius for better convergence.
                # Choose a direction arbitrarily ro be used for the inclusion constraint.
                (i,j) = (container1.task.predicate.source_agent,container1.task.predicate.target_agent)
                
                # Now impose the inclusion with consistency on the source and target direction for this edge.
                constraints += [get_inclusion_contraint(zeta = zeta, task_container= container1,source=i,target=j)]
                constraints += [get_inclusion_contraint(zeta = zeta, task_container= container2,source=i,target=j)]
                self._zeta_variables += [zeta]
                
        # 2) Eventually intersection constraints.
        maximal_sets = self.compute_maximal_sets_intersecting_eventually_tasks()
        for eventually_container,always_container_set in maximal_sets.items() :
            zeta = self._optimizer.variable(eventually_container.task.state_space_dimension)* self.communication_radius # scaled by the communication radius for better convergence.* self.communication_radius # scaled by the communication radius for better convergence.
            
            (i,j) = (eventually_container.task.predicate.source_agent,eventually_container.task.predicate.target_agent)
            
            if any_parametric(always_container_set.union({eventually_container})) : # check if there is any parametric task over which to impose the task.
                # Always tasks inclusion.
                for always_container in always_container_set :
                    constraints += [get_inclusion_contraint(zeta = zeta, task_container = always_container,source=i,target=j)]
                
                # Eventually tasks inclusion.
                constraints += [ get_inclusion_contraint(zeta = zeta, task_container =  eventually_container,source=i,target=j) ]
                self._zeta_variables += [zeta]
           
        return constraints   
    
    
    def compute_maximal_sets_intersecting_eventually_tasks(self) -> dict[TaskOptiContainer,set[TaskOptiContainer]]:
        
        # Separate always and eventually tasks.
        always_task_containers      : list[TaskOptiContainer]  = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,AlwaysOperator)]
        eventually_task_containers  : list[TaskOptiContainer]  = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,EventuallyOperator)]
        
        if len(eventually_task_containers) == 0:
            return {}
        
        # Find all always tasks that intersect a single eventually task.
        intersection_sets = {}
        for eventually_container in eventually_task_containers :
            intersecting_always_tasks = []
            for always_container in always_task_containers :
                if  not (eventually_container.task.temporal_operator.time_interval / always_container.task.temporal_operator.time_interval ).is_empty() :
                    intersecting_always_tasks == [always_container] 
            
            intersection_sets[eventually_container] = intersecting_always_tasks
            
            
        # For each eventually task compute the power set of always tasks that intersect it.
        power_sets  : dict[TaskOptiContainer,set[TaskOptiContainer]]  = { eventually_container : powerset(intersecting_always_tasks) for eventually_container,intersecting_always_tasks in intersection_sets.items() } 
        
        # Among the sets in the power set, find the sets with non empty intersection.
        candidate_sets = {}
        for eventually_container,always_container_set in power_sets.items() :
            # Compute the intersection of all tasks in this subset.
            intersection = TimeInterval(a = float("-inf"),b = float("inf"))
            candidate_sets[eventually_container]  = []
            
            for always_task in always_container_set :
                intersection = intersection / always_task.task.temporal_operator.time_interval
            
            # At last check intersection with the eventually task.
            intersection = intersection / eventually_container.task.temporal_operator.time_interval
            
            # If there is a nonzero intersection then add a common intersection constraint.
            if not intersection.is_empty() :
                candidate_sets[eventually_container] += [always_container_set] 
                
        # Obtain the maximal sets among the candidate intersecting sets.  
        maximal_sets = {}   
        for eventually_container,always_container_set in candidate_sets.items() :
            
            for set_i in always_container_set :
                is_subset = False
                for set_j in always_container_set :
                    
                    if set_i != set_j :
                        if set_i.issubset(set_j) :
                            is_subset = True
                            break
                        
                if not is_subset :
                    maximal_sets[eventually_container] = set_i
                            
        return maximal_sets  
            
  
    
    def setup_optimizer(self,num_iterations :int) :
        """
        setup the optimization problem for the given agent
        """    
        
        # If there is not prametric task you can skip
        if not self.is_computing_edge :
            return  
        # If there is not prametric task you can skip
        if not self.is_computing_edge :
            return  
        
        if self._is_initialized_for_optimization :
            message = "The optimization problem was already set up. You can only set up the optimization problem once at the moment."
            self._logger.error(message)
            raise RuntimeError(message)
        
        if len(self._task_containers) == 0 :
            message = "The agent does not have any tasks to decompose. The optimization problem will not be setup and solved."
            self._logger.warning(message)

        
        self._num_iterations     = int(num_iterations)     # number of iterations.
        self._decay_rate          = 0.3                    # decay rate of the learning rate (<1)
        self._is_initialized_for_optimization  = True
        
        if self._use_non_linear_cost:
            cost = 1
        else :
            cost = 0
        
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
                self._optimizer.set_initial(task_container.center_var, center_guess / task_container._center_scale_factor) # set an initial guess value for the center variable
                
                if self._use_non_linear_cost:
                    cost *= 1/task_container.scale_var
                else :
                    cost += -10*task_container.scale_var 
            
        cost += 500*self._penalty # setting cost toward zero
       
        # set up private and shared constraints (also if you don't have constraints this part will be printed)
        overloading_constraints   = self._compute_overloading_constraints()
        shared_path_constraint    = self._compute_shared_constraints()
        communication_constraints = self._compute_communication_consistentcy_constraints()
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
        
        # Problem options
        p_opts = dict(print_time=False, 
                      verbose=False,
                      expand=True,
                      compiler='shell',
                      jit=True,  # Enable JIT compilation
                      jit_options={"flags": '-O3', "verbose": True, "compiler": 'gcc'})  # Specify the compiler (optional, default is gcc))

        # Solver options
        s_opts = dict(
            print_level=1,
            tol=1e-6,
            max_iter=1000,
            )

        self._optimizer.solver("ipopt",p_opts,s_opts)
        self._cost = cost
        self._optimizer.solver("ipopt",p_opts,s_opts)
        self._cost = cost
        
        
    def solve_local_problem(self,print_result:bool=False) -> None :
        
        # skip computations and add to counter
        if not self.is_computing_edge :
        # skip computations and add to counter
        if not self.is_computing_edge :
            self._current_iteration += 1
            return  
            return  
            
        # Update the consensus parameters.
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                consensus_average = task_container.compute_consensus_average_parameter() 
                self._optimizer.set_value(task_container.average_consensus_param,   consensus_average )
                
        if self._warm_start_solution != None :
            self._optimizer.set_initial(self._warm_start_solution.value_variables())
         

        try :
            with contextlib.redirect_stdout(f):
                sol : ca.OptiSol = self._optimizer.solve() 
                # Reset the StringIO buffer
                f.seek(0)
                f.truncate(0)
                
        except  Exception as e:
            self._logger.error(f"An error occured while solving the optimization problem for agent {self._edge_id}. The error message is: {e}")
            self._logger.error(f"An error occured while solving the optimization problem for agent {self._edge_id}. The error message is: {e}")
            raise e
        
        if print_result :
            print("--------------------------------------------------------")
            print(f"Agent {self._edge_id } SOLUTION")
            print("--------------------------------------------------------")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    print("Center and Scale")
                    print(self._optimizer.value(task_container.center_var) * task_container._center_scale_factor)
                    print(self._optimizer.value(task_container.center_var) * task_container._center_scale_factor)
                    print(self._optimizer.value(task_container.scale_var))
                    for zeta in self._zeta_variables :
                        print("Zeta")
                        print(self._optimizer.value(zeta) * self.communication_radius)
                        print(self._optimizer.value(zeta) * self.communication_radius)
            
            print("penalty")
            print(self._optimizer.value(self._penalty))
            print("--------------------------------------------------------")
        
        self._penalty_values.append(self._optimizer.value(self._penalty))
        self._cost_values.append(self._optimizer.value(self._cost))
        self._warm_start_solution = sol
        
        # Update lagrangian coefficients
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                
                lagrangian_coefficient = self._optimizer.value(self._optimizer.dual(task_container.local_constraint))[:,np.newaxis]
                task_container.update_lagrangian_variable_from_self(lagrangian_variable=lagrangian_coefficient) # Save the lagrangian variable associated with this task.
        
        
        # Update current iteration count.
        self._current_iteration += 1
        
        # Set the optimizer to SQPMETHOD after the first solution is found. This method is much faster than Ipopt once a good initial solution is found.
        if self._switch_to_SQPMETHOD and (self._current_iteration==int(0.002*self._num_iterations)) :
            
            opts = {
                'expand': True,
                'print_time': False,         # Suppress CasADi print time
                'verbose': False,        # General silent option for sqpmethod
                'qpsol': 'osqp',         # Use the OSQP solver
                'qpsol_options': {
                    'verbose': False,
                    'print_time':0,
                    'print_problem': 0}     # Silence OSQP solver output
            }


            
            self._optimizer.solver("sqpmethod", opts)

        
    
    def step_consensus_variables_with_currently_available_lagrangian_coefficients(self) -> None :
        learning_rate = self._learning_rate_0 *(1/self._current_iteration*self._decay_rate)
        learning_rate = self._learning_rate_0 *(1/self._current_iteration*self._decay_rate)
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                for edge_id in task_container.neighbour_edges_id :
                    task_container.step_consensus_variable_from_self(neighbour_edge_id = edge_id,learning_rate = learning_rate)
                    
    
    # Callbacks and notifications.
    def update_consensus_variable_from_edge_neighbour_callback(self,consensus_variables_map_from_neighbour:dict[int,np.ndarray],neighbour_edge_id:int, parent_task_id:int) -> None :
        """Update the task of the agent"""
        self._logger.info(f"Receiving consensus variable from neighbour edge {neighbour_edge_id} for parent task {parent_task_id}")
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
        self._logger.info(f"Receiving lagrangian variable from neighbour edge {neighbour_edge_id} for parent task {parent_task_id}")
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                if task_container.parent_task_id == parent_task_id :
                    task_container.update_lagrangian_variable_from_neighbours(neighbour_edge_id = neighbour_edge_id,lagrangian_variable= lagrangian_variable_from_neighbour)
                    break
        
    
    def notify(self,event_type:str):
        if event_type == "new_consensus_variable" :
            self._logger.info(f"Notifying neighbours about the new consensus variable")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    for callback in self._subscribers[event_type] :
                        callback( task_container._consensus_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
                    
                    
        elif event_type == "new_lagrangian_variable" :
            self._logger.info(f"Notifying neighbours about the new lagrangian variable")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    
                    for callback in self._subscribers[event_type] :
                        callback( task_container._lagrangian_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
                        
                        
                        
    def deparametrize_container(self,task_container : TaskOptiContainer) -> StlTask | None :
        
        if task_container.task.is_parametric :
            try :
                center = np.asanyarray(self._optimizer.value(task_container.center_var)) * task_container._center_scale_factor
                center = np.asanyarray(self._optimizer.value(task_container.center_var)) * task_container._center_scale_factor
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
                                                   source_agent_id = task_container.task.predicate.source_agent, 
                                                   target_agent_id = task_container.task.predicate.target_agent)
            new_task = StlTask(predicate = new_predicate, temporal_operator = task_container.task.temporal_operator)
        else :
            # if task is not parametric return the task as it is
            new_task = task_container.task
        return new_task
        
    def get_parametric_container_with_parent_task(self,parent_task_id:int) -> TaskOptiContainer | None :
    def get_parametric_container_with_parent_task(self,parent_task_id:int) -> TaskOptiContainer | None :
        
        for task_container in self._task_containers :
            if task_container.task.is_parametric:
                if task_container.parent_task_id == parent_task_id :
                    return task_container
        return None
    
    
    
class EdgeComputingGraph(nx.Graph) :
    def __init__(self,incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
    
    def add_node(self, node_for_adding, **attr):
        """ Adds an edge to the graph."""
        super().add_node(node_for_adding, **attr)
        self._node[node_for_adding][AGENT] = EdgeComputingAgent(edge_id = node_for_adding)
    
    
    
def extract_computing_graph_from_communication_graph(communication_graph :CommunicationGraph, communication_radius:float = 1.0) :
    
    if not isinstance(communication_graph,CommunicationGraph) :
        raise ValueError("The input graph must be a communication graph")
    
    computing_graph = EdgeComputingGraph()
    
    if not nx.is_tree(communication_graph) :
        raise ValueError("The communication graph must be acyclic to obtain a valid computation graph")
    
    for edge in communication_graph.edges :
        computing_graph.add_node(edge_to_int(edge))
    
    for edge in communication_graph.edges :    
        # Get all edges connected to node1 and node2
        edges_node1 = set(communication_graph.edges(edge[0]))
        edges_node2 = set(communication_graph.edges(edge[1]))
    
        # Combine the edges and remove the original edge
        adjacent_edges = list((edges_node1 | edges_node2) - {edge, (edge[1],edge[0])})
        computing_edges = [ (edge_to_int(edge), edge_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    # set communication radius.
    for node in computing_graph.nodes :
        computing_graph.nodes[node][AGENT].communication_radius = communication_radius
    
    return computing_graph
    
    
    
          
def run_task_decomposition(communication_graph:nx.Graph,task_graph:nx.Graph, communication_radius:float ,number_of_optimization_iterations : int,logger_file:str = None,logger_level:int = logging.INFO) :
    
    
    
class EdgeComputingGraph(nx.Graph) :
    def __init__(self,incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
    
    def add_node(self, node_for_adding, **attr):
        """ Adds an edge to the graph."""
        super().add_node(node_for_adding, **attr)
        self._node[node_for_adding][AGENT] = EdgeComputingAgent(edge_id = node_for_adding)
    
    
    
def extract_computing_graph_from_communication_graph(communication_graph :CommunicationGraph, communication_radius:float = 1.0) :
    
    if not isinstance(communication_graph,CommunicationGraph) :
        raise ValueError("The input graph must be a communication graph")
    
    computing_graph = EdgeComputingGraph()
    
    if not nx.is_tree(communication_graph) :
        raise ValueError("The communication graph must be acyclic to obtain a valid computation graph")
    
    for edge in communication_graph.edges :
        computing_graph.add_node(edge_to_int(edge))
    
    for edge in communication_graph.edges :    
        # Get all edges connected to node1 and node2
        edges_node1 = set(communication_graph.edges(edge[0]))
        edges_node2 = set(communication_graph.edges(edge[1]))
    
        # Combine the edges and remove the original edge
        adjacent_edges = list((edges_node1 | edges_node2) - {edge, (edge[1],edge[0])})
        computing_edges = [ (edge_to_int(edge), edge_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    # set communication radius.
    for node in computing_graph.nodes :
        computing_graph.nodes[node][AGENT].communication_radius = communication_radius
    
    return computing_graph
    
    
    
          
def run_task_decomposition(communication_graph:nx.Graph,task_graph:nx.Graph, communication_radius:float ,number_of_optimization_iterations : int,logger_file:str = None,logger_level:int = logging.INFO) :
    """Task decomposition pipeline"""
   
    # set log level
    logging.basicConfig(level=logger_level)
    
    # Normalize the task graph and the communication graph to have the same nodes as a precaution.
    communication_graph,task_graph = normalize_graphs(communication_graph, task_graph)
    
   
    # set log level
    logging.basicConfig(level=logger_level)
    
    # Normalize the task graph and the communication graph to have the same nodes as a precaution.
    communication_graph,task_graph = normalize_graphs(communication_graph, task_graph)
    
    # Check the communication graph.
    if not nx.is_connected(communication_graph) :
        raise ValueError("The communication graph is not connected. Please provide a connected communication graph")
    if not nx.is_tree(communication_graph) :
        raise ValueError("The communication graph is not a acyclic. Please provide an acyclic communication graph")
    
    original_task_graph :TaskGraph = clean_task_graph(task_graph) # remove edges that do not have tasks to keep the graph clean..
    new_task_graph      :TaskGraph = create_task_graph_from_edges(communication_graph.edges) # base the new task graph on the communication graph.
    original_task_graph :TaskGraph = clean_task_graph(task_graph) # remove edges that do not have tasks to keep the graph clean..
    new_task_graph      :TaskGraph = create_task_graph_from_edges(communication_graph.edges) # base the new task graph on the communication graph.

    # Create computing graph. Communication radius important to ensure communication consistency of the solution (and avoid diverging iterates).
    computing_graph :EdgeComputingGraph = extract_computing_graph_from_communication_graph(communication_graph = communication_graph, communication_radius = communication_radius)
    # Create computing graph. Communication radius important to ensure communication consistency of the solution (and avoid diverging iterates).
    computing_graph :EdgeComputingGraph = extract_computing_graph_from_communication_graph(communication_graph = communication_graph, communication_radius = communication_radius)

    critical_task_to_path_mapping : dict[StlTask,list[int]] = {} # mapping of critical tasks to the path used to decompose them.
    
    # For each each edge check if there is a decomposition to be done.
    for edge in original_task_graph.edges : 
        
        if edge in communication_graph.edges :
            task_list = original_task_graph[edge[0]][edge[1]][MANAGER].tasks_list
            # add the tasks to the new task graph.
            
            
            new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task_list)
            
            # Add tasks to the computing graph to go on with the decomposition.
            for task in task_list :
                container = TaskOptiContainer(task = task) # non parametric task containers.
                computing_graph.nodes[edge_to_int((edge[1],edge[0]))][AGENT].add_task_containers(task_containers = container)
                container = TaskOptiContainer(task = task) # non parametric task containers.
                computing_graph.nodes[edge_to_int((edge[1],edge[0]))][AGENT].add_task_containers(task_containers = container)
        
        # if the edge is a critical edge
        elif not (edge in communication_graph.edges) :  # this edge is inconsistent with the communication graph
        elif not (edge in communication_graph.edges) :  # this edge is inconsistent with the communication graph
            
            # Retrive all the tasks on the edge because such tasks will be decomposed
            tasks_to_be_decomposed: list[StlTask] =  original_task_graph[edge[0]][edge[1]][MANAGER].tasks_list
            
            path = nx.shortest_path(communication_graph,source = edge[0],target = edge[1]) # find shortest path connecting the two nodes.
            edges_through_path = edge_set_from_path(path=path) # find edges along the path.
            
            for parent_task in tasks_to_be_decomposed : # add a new set of tasks for each edge along the path of the task to be decomposed
                
                critical_task_to_path_mapping[parent_task] = path # map the critical task to the path used to decompose it.
                if (parent_task.predicate.source_agent,parent_task.predicate.target_agent) != (path[0],path[-1]) : # case the direction is incorrect.
                    parent_task.flip() # flip the task such that all tasks have the same direction of the node.
                
                for source_node,target_node in  edges_through_path :
                    
                    # Create parametric task along this edge and add it to the communication graph.
                    subtask = create_parametric_collaborative_task_from(task = parent_task , source_agent_id = source_node, target_agent_id = target_node ) 
                    
                    
                    # Create task container to be given to the computing agent corresponding to the edge.
                    task_container = TaskOptiContainer(task=subtask, center_scale_factor = communication_radius )
                    task_container = TaskOptiContainer(task=subtask, center_scale_factor = communication_radius )
                    task_container.set_parent_task_and_decomposition_path(parent_task = parent_task,decomposition_path = path)
                    computing_graph.nodes[edge_to_int((source_node,target_node))][AGENT].add_task_containers(task_containers = task_container)
        
    
    number_of_optimization_iterations = int(number_of_optimization_iterations)
                    computing_graph.nodes[edge_to_int((source_node,target_node))][AGENT].add_task_containers(task_containers = task_container)
        
    
    number_of_optimization_iterations = int(number_of_optimization_iterations)
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for node in computing_graph.nodes :
        computing_graph.nodes[node][AGENT].setup_optimizer(num_iterations = number_of_optimization_iterations)
    
    
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
    for jj in tqdm(range(number_of_optimization_iterations )) :
    for jj in tqdm(range(number_of_optimization_iterations )) :
        
        # Everyone shares the current consensus variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].notify("new_consensus_variable")
            
        # Everyone solves the local problem and updates the lagrangian variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].solve_local_problem(  print_result=False )
            computing_graph.nodes[node][AGENT].solve_local_problem(  print_result=False )
            
        # Everyone updates the value of the consensus variables.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].notify("new_lagrangian_variable")
        
        # Everyone updates the consensus variables using the lagrangian coefficients updates.
        for node in computing_graph.nodes :
            computing_graph.nodes[node][AGENT].step_consensus_variables_with_currently_available_lagrangian_coefficients()
    
    
    
    # Extract solution.   
    fig,(ax1,ax2) = plt.subplots(1,2)
    for edge_id in computing_graph.nodes :
        agent = computing_graph.nodes[edge_id][AGENT]
        
        if agent.is_initialized_for_optimization :
            ax1.plot(range( number_of_optimization_iterations),agent._penalty_values,label=f"Agent :{edge_id}")
            ax1.set_ylabel("penalties")
            
            ax2.plot(range( number_of_optimization_iterations),agent._cost_values,label=f"Agent :{edge_id}")
            ax2.set_ylabel("cost")
    
    ax1.legend()
    ax2.legend()
    
    # Fill the new_task graph.
    for edge in new_task_graph.edges :
        computing_edge_id = edge_to_int(edge)
        
        for container in computing_graph.nodes[computing_edge_id][AGENT].task_containers :
            task = computing_graph.nodes[computing_edge_id][AGENT].deparametrize_container(task_container = container) # returns the task perse if there is nothing to deparatmetrize
            new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task) # add the task to the new task graph once the solution is found.
            task = computing_graph.nodes[computing_edge_id][AGENT].deparametrize_container(task_container = container) # returns the task perse if there is nothing to deparatmetrize
            new_task_graph[edge[0]][edge[1]][MANAGER].add_tasks(task) # add the task to the new task graph once the solution is found.
    
    new_task_graph = clean_task_graph(new_task_graph) # clean the graph from edges without any tasks.
    
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
                center = -np.asarray(agent.optimizer.value(task_container.center_var)).flatten()*task_container._center_scale_factor
                scale  = float(agent.optimizer.value(task_container.scale_var))
            else :
                center = np.asarray(agent.optimizer.value(task_container.center_var)).flatten()*task_container._center_scale_factor
                scale  = float(agent.optimizer.value(task_container.scale_var))
            task_container    = agent.get_parametric_container_with_parent_task(parent_task_id = critical_task.task_id)
            task = task_container.task
            
            if task.predicate.source_agent != edge[0] :
                center = -np.asarray(agent.optimizer.value(task_container.center_var)).flatten()*task_container._center_scale_factor
                scale  = float(agent.optimizer.value(task_container.scale_var))
            else :
                center = np.asarray(agent.optimizer.value(task_container.center_var)).flatten()*task_container._center_scale_factor
                scale  = float(agent.optimizer.value(task_container.scale_var))
            
            center_sum += center
            scale_sum  += scale
        
        print(f"Number of hyperplanes  : {len(critical_task.predicate.A)}")
        print(f"Temporal operator      : {critical_task.temporal_operator}")
        print(f"Original edge          : {(critical_task.predicate.source_agent,critical_task.predicate.target_agent)}")   
        print(f"Decomposition path     : {path}")
        print(f"Original center        : {critical_task.predicate.center.flatten()}")
        print(f"Original center        : {critical_task.predicate.center.flatten()}")
        print(f"Center summation       : {center_sum}")  
        print(f"Decomposition accuracy : {scale_sum}")
    
    
    return new_task_graph,computing_graph
    return new_task_graph,computing_graph




    
# def printDecompositionResult(decompositionPaths : list[list[int]], decompositionAgents : dict[int,EdgeComputingAgent], edge_objectsLists : list[GraphEdge]) :
   
   
#     # now we need to extract informations for each task 
#     decompositionDictionay : dict[TaskOptiContainer,dict[tuple[int,int],TaskOptiContainer]]= {}
    
#     # select the agents that have actually made computations 
#     computingAgents = {agentID:agent for agentID,agent in decompositionAgents.items() if agent.is_initialized_for_optimization}
    
    
#     for path in decompositionPaths :
#         # now take initial and final index since this is the non communicating edge 
#         (i,j) = path[0],path[-1]
#         edge_obj = findEdge(edge_list = edge_objectsLists, edge =(i,j)) # get the corresponding edge
#         decomposedTasks : list[TaskOptiContainer] = edge_obj.task_containers                                # get all the tasks defined on this edge
        
#         for task_container in decomposedTasks :
#             tasksAlongThePath = {}
#             for agentID,agent in computingAgents.items() :
#                 agenttask_containers = agent.taskConstainers # search among the containers you optimised for 
#                 for container in agenttask_containers : # check which task on the agent you have correspondance to
#                     if task_container.taskID == container.taskID :
#                         tasksAlongThePath[agent] = container # add the partial task now
            
#             decompositionDictionay[task_container] = tasksAlongThePath  # now you have dictionary with parentTask and another dictionary will all the child tasks       
    
    
#     if len(decompositionDictionay) <= 3 and len(decompositionDictionay)>1 :
#         fig,ax = plt.subplots(len(decompositionDictionay))
#     elif len(decompositionDictionay)==1 :
#         fig,ax = plt.subplots()
#         ax = [ax]
#     else :
#         rows = -(-len(decompositionDictionay)//3) # ceil
#         fig,ax = plt.subplots(rows,3)
#         ax = ax.flatten()
    
#     counter = 0    
#     for decomposedTaskContainer,tasksAlongPath in decompositionDictionay.items() :
        
#         bprime = decomposedTaskContainer.task.predicate.b +  decomposedTaskContainer.task.predicate.A@decomposedTaskContainer.task.predicate.center
#         poly.Polytope(A = decomposedTaskContainer.task.predicate.A,b= bprime).plot(ax=ax[counter],color="red",alpha=0.3)
#         ax[counter].quiver([0],[0],decomposedTaskContainer.task.predicate.center[0],decomposedTaskContainer.task.predicate.center[1],scale_units='xy',angles='xy', scale=1)
        
#         ax[counter].set_xlim([-40,40])
#         ax[counter].set_ylim([-40,40])
        
#         print("**************************************************************")
#         print(f"TASK ID           : {decomposedTaskContainer.task.task_id}")
#         print(f"Temporal operator : {decomposedTaskContainer.task.temporal_operator}")
#         print(f"Original Edge     : {(decomposedTaskContainer.parent_task.source_agent, decomposedTaskContainer.parent_task.target_agent)}")
#         print(f"Original Center   : {decomposedTaskContainer.parent_task.target_agent}")
#         print("**********************Found Solution**************************")
        
#         # sum of alpha values 
#         scaleSums      = 0 
#         centerSums     = np.array([0.,0.])
#         previousCenter = np.array([0.,0.])
#         yaverageSum    = 0
        
#         for computingAgent, childTaskContainer in tasksAlongPath.items() :
             
#             # the computing agent is at one of the the nodes of the edge
#             scaleSums  += computingAgent.optimizer.value(childTaskContainer.task.scale)
#             centerSums += computingAgent.optimizer.value(childTaskContainer.task.center)
#             yaverageSum += childTaskContainer.computeconsensusAverage()
            
#             # print successive edge sets
#             bprime = scaleSums*decomposedTaskContainer.task.predicate.b +  decomposedTaskContainer.task.predicate.A@centerSums
#             poly.Polytope(A = decomposedTaskContainer.task.predicate.A,b= bprime).plot(ax=ax[counter],color="blue",alpha=0.3)
#             ax[counter].quiver(previousCenter[0],previousCenter[1],centerSums[0]-previousCenter[0],centerSums[1]-previousCenter[1],angles='xy', scale_units='xy', scale=1)
            
#             previousCenter += computingAgent.optimizer.value(childTaskContainer.task.center)
            
            
            
#         print(f"Sum of scales         : {scaleSums}".ljust(30)      +     "(<= 1.)")
#         print(f"Sum of Centers        : {centerSums}".ljust(30)     +     "(similar to original)")
#         print(f"consensus variables   : {np.sum(yaverageSum)}".ljust(30) + "(required 0.0)")
#         print("**************************************************************")
#         ax[counter].set_title(f"Task edge {decomposedTaskContainer.task.sourceTarget}\n Operator {decomposedTaskContainer.task.temporal_operator} {decomposedTaskContainer.task.time_interval}")
#         ax[counter].grid(visible=True)
        
#         counter+=1
    
#     for jj in range(len(decompositionDictionay),len(ax)): # delete useless axes
#         fig.delaxes(ax[jj])
     
   


    
if __name__== "__main__" :
    
    
    # Test power set
    print("Testing powerset")
    print(list(powerset({1,2,3,4,5})))
    
    # Test Task container
    print("Testing Task Container")
    i = 3
    j = 2
    A,b = regular_2D_polytope(number_hyperplanes=3,distance_from_center=2)
    t_operator = AlwaysOperator(time_interval = TimeInterval(a = 0,b = 10))
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
    
    
    
    