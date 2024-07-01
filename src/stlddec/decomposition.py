import numpy as np
import casadi as ca
from itertools import chain,combinations
from .stl_task import * 
from .transport import Publisher
from .graphs import *
import networkx as nx 
import matplotlib.pyplot as plt 
import logging
import polytope as poly



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
    
    def __init__(self, task : StlTask) -> None:
        
        """
        """    
         
        self._task = task
        
        # These attributes are enables only when the task is parametric
        self._parent_task            :"StlTask"            = None  # instance of the parent class
        self._decomposition_path     :list[int]            = []    # path of the decomposition of the task when the task is parametric
        self._neighbour_edges_id     :list[int]            = []    # list of the edges neighbouring to the task edge and expressed as a single id. Ex : (4,1)-> (41) 
        
        if isinstance(self.task.predicate,CollaborativePredicate):
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
            
        
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
        self._consensus_average_param     : ca.MX = None # average consensus parameter for the concesusn variable y
        
    
    @property
    def has_parent_task(self):
        return self._parent_task is not None
    
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
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
        return self._center_var
    
    @property
    def scale_var(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
        return self._scale_var
    
    @property
    def eta_var(self):
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric. No scale variable is available")
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
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
    def consensus_average_param(self):
        return self._consensus_average_param
    
    
    def set_optimization_variables(self,opti:ca.Opti):
        """Set the optimization variables for the task container"""
        if not self.task.is_parametric :
            raise ValueError("The task is not parametric hence the optimization variables cannot be set")
        
        self._center_var = opti.variable(self.task.predicate.state_space_dim,1)                            # center the parameteric formula
        self._scale_var  = opti.variable(1)                                                         # scale for the parametric formula
        self._eta_var     = ca.vertcat(self._center_var,self._scale_var)                                    # optimization variable
        
        coupling_constraint_size      = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        self._average_consensus_param = opti.parameter(coupling_constraint_size,1) # average consensus parameter for the concesusn variable y
        self._is_initialized          = True

    
    def compute_consensus_average_parameter(self,) -> np.ndarray :
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
    
    def set_parent_task_and_decomposition_path(self,parent_task:"StlTask",decomposition_path:list[int])-> None:
        
        if not self._task.predicate.is_parametric :
            raise Warning("The task is not parametric. The parent task and decomposition path setting will be ignored")
        else :
            self._parent_task = parent_task
            self._decomposition_path = decomposition_path
            
            for edge in decomposition_path :
                if ( (edge[0] == self.task.predicate.source_agent) or
                     (edge[1] == self.task.predicate.source_agent) or 
                     (edge[0] == self.task.predicate.target_agent) or 
                     (edge[1] == self.task.predicate.target_agent) ) :
                    
                    self._neighbour_edges_id.append(tuple_to_int(edge))
                    
        # Initialize consensus parameters and lagrangian multiplies         
        for edge in self._neighbour_edges_id :
            self._consensus_param_neighbours_from_neighbours[edge]   = np.zeros((self._coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ij} in paper)
            self._consensus_param_neighbours_from_self[edge]         = np.zeros((self._coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ji} in paper)
            self._lagrangian_param_neighbours_from_neighbours[edge]  = np.zeros((self._coupling_constraint_size,1)) # lagrangian_coefficient from neighbur 
    
    
    def set_task_constraint_expression(self, constraint : ca.MX) -> None :
        """Save the constraint for later retrieval of the lagrangian multipliers"""
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
    """Return the constraint for the inclusion of a polytope in another polytope"""
    
    # Only collaborative parametric task check.
    if not isinstance(task_container.task.predicate,CollaborativePredicate) :
        raise ValueError("Only parametric collaborative tasks are allowed to be decomposed. No current support for Individual tasks")
    elif not task_container.task.is_parametric :
        raise ValueError("Only parametric tasks are allowed to be decomposed.")
    
    # A @ (x_i-x_j - c) <= b   =>  A @ (e_ij - c) <= b 
    # becomes 
    # A @ (x_j-x_i + c) >= -b   =>  -A @ (e_ji + c) <= b  =>  A_bar @ (e_ji - c_bar) <= b   (same sign of b,A change the sign)
        
    # Check the target source pairs as the inclusion relation needs to respect the direction of the task.
    if (source == task_container.task.predicate.source_agent) and (target == task_container.task.predicate.target_agent) :
        A = task_container.task.predicate.A
        b = task_container.task.predicate.b
        A_z = np.hstack((A,b)) 
        constraints = A@zeta - A_z@task_container.eta_var <= 0 
    
    else :
        
        A = -task_container.task.predicate.A   
        b =  task_container.task.predicate.b
        A_z = np.hstack((A,b)) 
        
        # The flip matrix reverts the sign of the center variable and lets the scale factor with the same sign.
        flip_mat = -np.eye(task_container.task.state_space_dimension)
        flip_mat[-1,-1] = 1
        
        constraints = A@zeta - A_z@flip_mat@task_container.eta_var <= 0 

    return constraints



class EdgeComputingAgent(Publisher) :
    """
       Edge Agent Task Decomposition
    """
    def __init__(self,edge : tuple[UniqueIdentifier,UniqueIdentifier]) -> None:
        """
        Args:
            agentID (int): Id of the agent
        """        
        
        super().__init__()
        
        self._optimizer         : ca.Opti          = ca.Opti() # optimizer for the single agent.
        self._edge              : int              = edge 
        self._edge_id           : int              = tuple_to_int(edge)
        self._task_containers   : list[TaskOptiContainer] = [] # list of task containers.
        self._is_computing_edge : bool = False
       
        self._warm_start_solution : ca.OptiSol   = None
        self._was_solved_already_once = False

        self._num_iterations  = None
        self._current_iteration = 0
        
        
        self._penalty = self._optimizer.variable(1)
        self._optimizer.subject_to(self._penalty>=0)
        self._optimizer.set_initial(self._penalty,40)
        
        self._penalty_values = []
        self._cost_values    = []
        self._is_initialized_for_optimization = False
        self._zeta_variables = []
        
        self.add_topic("new_consensus_variable")
        self.add_topic("new_lagrangian_variable")
        
    @property
    def edge(self):
        return self._edge 
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
        return self._is_computing_edge
    
    def add_task_container(self,task_container:TaskOptiContainer) :
 
        # Checks.
        if not isinstance(task_container.task.predicate,CollaborativePredicate) :
            raise ValueError("Only collaborative tasks are allowed as these ar the only ones that should be decomposed.")
        
        if task_container.task.is_parametric and not task_container.has_parent_task :
            raise ValueError("The task container provided contains a parameteric task but does not have an assigned parent task. You must set the parent task together with the decomposition path using the method set_parent_task_and_decomposition_path")
                
        # Create the optimization variables for the solver.
        if task_container.task.is_parametric:
            task_container.set_optimization_variables(self._optimizer)
            
        self._task_containers.append(task_container)
            
    def _compute_shared_constraints(self) -> list[ca.MX]:
        """computes the shared inclusion constraint for the given agent. The shared constraints are the incluson of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints (list[ca.Function]): set of constraints
        """        
        
        constraints_list = []
        for container in self._task_containers :
            task                 = container.task                     # extract task
            num_computing_agents = container.length_decomposition_path -1  # Number of agents sharing the computation for this constraint
            
            # Compute constraints.
            M,Z  = get_M_and_Z_matrices_from_inclusion(P_including=container.parent_task, P_included=task) # get the matrices for the inclusion constraint
            Z    = Z/num_computing_agents # scale the Z matrix by the number of computing agents 
            constraint = (M@task.eta_var - Z)  - self._penalty + container.consensus_average_param <= 0 # set the constraint
            constraints_list = [constraint]
            
            container.set_task_constraint_expression(constraint=constraint)
        
        return constraints_list
        
     
            
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
            if not (container1.task.temporal_operator.time_interval / container2.task.temporal_operator.time_interval ).is_empty() : # there is an intersection
                
                
                zeta = self._optimizer.variable(container1.task.state_space_dimension)
                
                constraints += [get_inclusion_contraint(zeta = zeta, task_container= container1)]
                constraints += [get_inclusion_contraint(zeta = zeta, task_container= container1)]
                self._zeta_variables += [zeta]
                
        # 2) Eventually intersection constraints.
        maximal_sets = self.compute_maximal_sets_intersecting_eventually_tasks()
        for eventually_container,always_container_set in maximal_sets.items() :
            zeta = self._optimizer.variable(eventually_container.task.state_space_dimension)
        
            # Always formulas intersection.
            for always_container in always_container_set :
                constraints += [get_inclusion_contraint(zeta = zeta, task_container = always_container)]
        
        
            constraints += [ get_inclusion_contraint(zeta = zeta, task_container =  eventually_container) ]
            self._zeta_variables += [zeta]
           
        return constraints   
    
    
    def compute_maximal_sets_intersecting_eventually_tasks(self) -> dict[TaskOptiContainer,set[TaskOptiContainer]]:
        
        
        # Separate always and eventually tasks.
        always_task_containers = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,AlwaysOperator)]
        eventually_task_containers = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,EventuallyOperator)]
        
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
        power_sets     = { eventually_container : powerset(intersecting_always_tasks) for eventually_container,intersecting_always_tasks in intersection_sets.items() } 
        
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
        
        self._is_computing_edge = bool(len(self._task_containers))
        
        if not self._is_computing_edge :
            raise ValueError(f"Only computing agents can set up the optimization problem including their constraints and cost function. The current agent with assigned edge {self._edge} is not a computing agent.")
        
        if self._is_initialized_for_optimization :
            raise NotImplementedError("The optimization problem was already set up. You can only set up the optimization problem once at the moment.")
        
        if len(self._task_containers)==0 :
            raise Warning("The agent does not have any tasks to decompose. The optimization problem will not be set up")

        
        self._num_iterations     = int(num_iterations) # number of iterations.
        self._is_initialized_for_optimization  = True
        cost = 0
        
        for task_container in self._task_containers :
            
            if task_container.task.is_parametric:
                task_container.set_optimization_variables(self._optimizer) # creates scale,center and consensus variables for each task.
        
                # Set the scale factor positive in constraint
                self._optimizer.subject_to(task_container.scale_var >0)
                self._optimizer.set_initial(task_container.scale_var,0.2) # set an initial guess value
                cost += -task_container.scale_var
            
            cost += 12*self._penalty # setting cost toward zero
       
        # set up private and shared constraints (also if you don't have constraints this part will be printed)
        overloading_constraints = self._compute_overloading_constraints()
        shared_path_constraint   = self._compute_shared_constraints()
        
        if len(overloading_constraints) != 0:
            self._optimizer.subject_to(overloading_constraints ) # it can be that no private constraints are present
        if len(shared_path_constraint)!=0 :
            self._optimizer.subject_to(shared_path_constraint)
        
        
        print("-----------------------------------")
        print(f"Computing edge                     : {self._edge}")
        print(f"Number of overloading_constraints  : {len(overloading_constraints)}")
        print(f"Number of shared constraints       : {len(shared_path_constraint )}")
        print(f"Number of variables                : {len(self.optimizer.nx)}")
        
        # set up cost and solver for the problem
        self._optimizer.minimize(cost)
        p_opts = dict(print_time=False, verbose=False,expand = True)
        s_opts = dict(print_level=1)

        self._optimizer.solver("ipopt",p_opts,s_opts)
        self._cost = cost
        
        
    def solve_local_problem(self,print_result:bool=False) -> None :
        
        # Update the consensus parameters.
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                consensus_average = task_container.compute_consensus_average_parameter() 
                self._optimizer.set_value(task_container.consensus_average_param,   consensus_average )
                
        if self._warm_start_solution != None :
            self._optimizer.set_initial(self._warm_start_solution.value_variables())
        
        try :
            sol : ca.OptiSol = self._optimizer.solve() # the end 
        except  Exception as e: # work on python 3.x
            print("******************************************************************************")
            logging.error(f'The optimization for agent {self._edge } failed with output: %s', e)
            print("******************************************************************************")
            print("The latest value for the variables was the following : ")
            print("penalty                 :",self._optimizer.debug.value(self._penalty))
            print("cost                    :",self._optimizer.debug.value(self._cost))
            print("******************************************************************************")
            
            exit()
      
        
        if print_result :
            print("--------------------------------------------------------")
            print(f"Agent {self._edge } SOLUTION")
            print("--------------------------------------------------------")
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    print("Center and Scale")
                    print(self._optimizer.value(task_container.center_var))
                    print(self._optimizer.value(task_container.scale_var))
            
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
        
        # Set the optimizer to SQPMETHOD after the first solution is found. This method is much faster than Ipopt once a good initial solution is found.
        if not self._was_solved_already_once :
            self._was_solved_already_once = True
            self._optimizer.solver("sqpmethod",{"qpsol":"qpoases"}) # Much faster after an initial solution is found.
            
        # Update current iteration count.
        self._current_iteration += 1
    
    
    def update_consensus_variable_from_edge_neighbour_callback(self,consensus_variables_map_from_neighbour:dict[int,np.ndarray],neighbour_edge_id:int, parent_task_id:int) -> None :
        """Update the task of the agent"""
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                if task_container.parent_task_id == parent_task_id : # you have a task connected to the same parent id.
                    
                    # From the variables of the neighbur extract the one correspoding to your edge.
                    consensus_variable = consensus_variables_map_from_neighbour[self._edge_id]
                    # Update the consensus variables as received from the neighbour.
                    task_container.update_consensus_variable_from_neighbours(edge = neighbour_edge_id , consensus_variable= consensus_variable)
                    break
        
    def update_lagrangian_variable_from_edge_neighbour_callback(self,lagrangian_variable_from_neighbour:np.ndarray,neighbour_edge_id:int,parent_task_id:int) -> None :
        """Update the task of the agent"""
        for task_container in self._task_containers :
            if task_container.task.is_parametric :
                if task_container.parent_task_id == parent_task_id :
                    task_container.update_lagrangian_variable_from_neighbours(edge = neighbour_edge_id,lagrangian_variable= lagrangian_variable_from_neighbour)
                    break
        
    
    def notify(self,event_type:str):
        if event_type == "new_consensus_variable" :
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    
                    for callback in self._subscribers[event_type] :
                        callback( task_container._consensus_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
                    
                    
        elif event_type == "new_lagrangian_variable" :
            
            for task_container in self._task_containers :
                if task_container.task.is_parametric :
                    
                    for callback in self._subscribers[event_type] :
                        callback( task_container._lagrangian_param_neighbours_from_self , self._edge_id , task_container.parent_task_id )
            
          
#!Currently working here
def run_task_decomposition(edge_map: UndirectedEdgeMapping[GraphEdge]) :
    """Task decomposition pipeline"""
            
    # Create communication graph.
    comm_graph          :nx.Graph = create_communication_graph_from_edges(edge_map=edge_map)
    # Create task graph.
    original_task_graph :nx.Graph = create_task_graph_from_edges(edge_map = edge_map)
    
    if not nx.is_connected(comm_graph) :
        raise ValueError("The communication graph is not connected. Please provide a connected communication graph")
    if not nx.is_tree(comm_graph) :
        raise ValueError("The communication graph is not a acyclic. Please provide an acyclic communication graph")
    
    # Create computing graph (just used for plotting).
    computing_graph :nx.Graph = create_computing_graph_from_communication_graph(comm_graph = comm_graph)
    
    
    # Create the agents for decomposition
    decomposition_agents : dict[int,EdgeComputingAgent]= {}
    
    # crate a computing node for each node in the graph
    for edge in comm_graph.edges :
        decomposition_agents[tuple_to_int(edge)] = EdgeComputingAgent(edge=edge)
    
    path_list : list[int] =  []
    
    # For each each edge check if there is a decomposition to be done.
    for edge_obj in edge_map.values() :    
        if edge_obj.is_communicating and edge_obj.has_specifications : # for all the cummincating agent start putting the tasks over the nodes
            for task in edge_obj.tasks_list :
                decomposition_agents[edge_obj.edge].add_task(task)
        
        if (not edge_obj.is_communicating) and (edge_obj.has_specifications) : # Decomposition needed.
            
            # Retrive all the tasks on the edge because such tasks will be decomposed
            tasks_to_be_decomposed: list[StlTask] = edge_obj.tasks_list
            
            path = nx.shortest_path(comm_graph,source=edge_obj.edge[0],target = edge_obj.edge[1]) # find shortest path connecting the two nodes.
            path_list.append(path)# add path to the path list.
            edges_through_path = edge_set_from_path(path=path) # find edges along the path.
            
            for parent_task in tasks_to_be_decomposed : # add a new set of tasks for each edge along the path of the task to be decomposed
                
                
                if (parent_task.predicate.source_agent,parent_task.predicate.target_agent) != (path[0],path[-1]) : # case the direction is incorrect.
                    parent_task.flip() # flip the task such that all tasks have the same direction of the node.
                
                for source_node,target_node in  edges_through_path : # The edges here are directed.
                    
                    # Create parametric task along this edge.
                    subtask = create_parametric_collaborative_task_from(task = parent_task , source_agent_id = source_node, target_agent_id =target_node, decomposition_path = path ) 
                    edge_obj_subtask = edge_map[(source_node,target_node)]
                    edge_obj_subtask.add_tasks(subtask) # for the record add the task to the edge object
                    agent = decomposition_agents[(source_node,target_node)].add_task(subtask)
                    agent.add_task(task)
    
    number_of_optimization_iterations = 1000
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for agent in decomposition_agents.values() :
        agent.setup_optimizer(num_iterations = number_of_optimization_iterations)
    
    # Set the callback connections.
    agents_combinations = list(combinations(decomposition_agents.values(),2))
    for agent_i,agent_j in agents_combinations :
        if agent_i.edge != agent_j.edge :
            if agent_j.edge_id in computing_graph.neighbors(agent_i.edge_id) :
                agent_i.subscribe("new_consensus_variable",agent_j.update_consensus_variable_from_edge_neighbour_callback)
                agent_i.subscribe("new_lagrangian_variable",agent_j.update_lagrangian_variable_from_edge_neighbour_callback)
                
                agent_j.subscribe("new_consensus_variable",agent_i.update_consensus_variable_from_edge_neighbour_callback)
                agent_j.subscribe("new_lagrangian_variable",agent_i.update_lagrangian_variable_from_edge_neighbour_callback)
            
             
    # Solution loop.
    for jj in range(number_of_optimization_iterations ) :
        
        # Everyone shares the current consensus variables.
        for agent in decomposition_agents.values() :
            agent.notify("new_consensus_variable")
            
        # Everyone solves the local problem and updates the lagrangian variables.
        for agent in decomposition_agents.values() :
            agent.solve_local_problem(  print_result=False )
        
        # Everyone notifies about the new lagrangian coefficient found.
        for agent in decomposition_agents.values() :
            agent.notify("new_consensus_variable")
        
        
        # Everyone updates the value of the consensus variables.
        for agent in decomposition_agents.values() :
            agent.notify("new_lagrangian_variable")
        
        
        
    #! come back from here.
    fig,(ax1,ax2) = plt.subplots(1,2)
    for agentID,agent in decompositionAgents.items() :
        if agent.is_initialized_for_optimization :
            ax1.plot(range(num_optimization_iterations),agent._penaltyValues,label=f"Agent :{agentID}")
            ax1.set_ylabel("penalties")
            
            ax2.plot(range(num_optimization_iterations),agent._cost_values,label=f"Agent :{agentID}")
            ax2.set_ylabel("cost")
    
    ax1.legend()
    ax2.legend()
    printDecompositionResult(decompositionPaths =path_list, decompositionAgents =decompositionAgents, edge_objectsLists = edge_list)
    
    # now that the tasks have been decomposed it is time to clean the old tasks and construct the new taskGraph
    for edge_obj in edge_list :
        if (not edge_obj.is_communicating) and (edge_obj.hasSpecifications) :
            edge_obj.cleanTasks()     # after you have decomposed the tasks you can just clean them
    
    finalTaskGraph : nx.Graph = createTaskGraphFromEdges(edge_list=edge_list)
    
    nodesAttributes = deperametrizeTasks(decompostionAgents=decompositionAgents)
 
    # set the attributes
    nx.set_node_attributes(finalTaskGraph,nodesAttributes)
    
    return comm_graph,finalTaskGraph,original_task_graph


    
def printDecompositionResult(decompositionPaths : list[list[int]], decompositionAgents : dict[int,AgentTaskDecomposition], edge_objectsLists : list[GraphEdge]) :
   
   
    # now we need to extract informations for each task 
    decompositionDictionay : dict[TaskOptiContainer,dict[tuple[int,int],TaskOptiContainer]]= {}
    
    # select the agents that have actually made computations 
    computingAgents = {agentID:agent for agentID,agent in decompositionAgents.items() if agent.is_initialized_for_optimization}
    
    
    for path in decompositionPaths :
        # now take initial and final index since this is the non communicating edge 
        (i,j) = path[0],path[-1]
        edge_obj = findEdge(edge_list = edge_objectsLists, edge =(i,j)) # get the corresponding edge
        decomposedTasks : list[TaskOptiContainer] = edge_obj.task_containers                                # get all the tasks defined on this edge
        
        for task_container in decomposedTasks :
            tasksAlongThePath = {}
            for agentID,agent in computingAgents.items() :
                agenttask_containers = agent.taskConstainers # search among the containers you optimised for 
                for container in agenttask_containers : # check which task on the agent you have correspondance to
                    if task_container.taskID == container.taskID :
                        tasksAlongThePath[agent] = container # add the partial task now
            
            decompositionDictionay[task_container] = tasksAlongThePath  # now you have dictionary with parentTask and another dictionary will all the child tasks       
    
    
    if len(decompositionDictionay) <= 3 and len(decompositionDictionay)>1 :
        fig,ax = plt.subplots(len(decompositionDictionay))
    elif len(decompositionDictionay)==1 :
        fig,ax = plt.subplots()
        ax = [ax]
    else :
        rows = -(-len(decompositionDictionay)//3) # ceil
        fig,ax = plt.subplots(rows,3)
        ax = ax.flatten()
    
    counter = 0    
    for decomposedTaskContainer,tasksAlongPath in decompositionDictionay.items() :
        
        bprime = decomposedTaskContainer.task.predicate.b +  decomposedTaskContainer.task.predicate.A@decomposedTaskContainer.task.predicate.center
        poly.Polytope(A = decomposedTaskContainer.task.predicate.A,b= bprime).plot(ax=ax[counter],color="red",alpha=0.3)
        ax[counter].quiver([0],[0],decomposedTaskContainer.task.predicate.center[0],decomposedTaskContainer.task.predicate.center[1],scale_units='xy',angles='xy', scale=1)
        
        ax[counter].set_xlim([-40,40])
        ax[counter].set_ylim([-40,40])
        
        print("**************************************************************")
        print(f"TASK ID           : {decomposedTaskContainer.task.task_id}")
        print(f"Temporal operator : {decomposedTaskContainer.task.temporal_operator}")
        print(f"Original Edge     : {(decomposedTaskContainer.parent_task.source_agent, decomposedTaskContainer.parent_task.target_agent)}")
        print(f"Original Center   : {decomposedTaskContainer.parent_task.target_agent}")
        print("**********************Found Solution**************************")
        
        # sum of alpha values 
        scaleSums      = 0 
        centerSums     = np.array([0.,0.])
        previousCenter = np.array([0.,0.])
        yaverageSum    = 0
        
        for computingAgent, childTaskContainer in tasksAlongPath.items() :
             
            # the computing agent is at one of the the nodes of the edge
            scaleSums  += computingAgent.optimizer.value(childTaskContainer.task.scale)
            centerSums += computingAgent.optimizer.value(childTaskContainer.task.center)
            yaverageSum += childTaskContainer.computeconsensusAverage()
            
            # print successive edge sets
            bprime = scaleSums*decomposedTaskContainer.task.predicate.b +  decomposedTaskContainer.task.predicate.A@centerSums
            poly.Polytope(A = decomposedTaskContainer.task.predicate.A,b= bprime).plot(ax=ax[counter],color="blue",alpha=0.3)
            ax[counter].quiver(previousCenter[0],previousCenter[1],centerSums[0]-previousCenter[0],centerSums[1]-previousCenter[1],angles='xy', scale_units='xy', scale=1)
            
            previousCenter += computingAgent.optimizer.value(childTaskContainer.task.center)
            
            
            
        print(f"Sum of scales         : {scaleSums}".ljust(30)      +     "(<= 1.)")
        print(f"Sum of Centers        : {centerSums}".ljust(30)     +     "(similar to original)")
        print(f"consensus variables   : {np.sum(yaverageSum)}".ljust(30) + "(required 0.0)")
        print("**************************************************************")
        ax[counter].set_title(f"Task edge {decomposedTaskContainer.task.sourceTarget}\n Operator {decomposedTaskContainer.task.temporal_operator} {decomposedTaskContainer.task.time_interval}")
        ax[counter].grid(visible=True)
        
        counter+=1
    
    for jj in range(len(decompositionDictionay),len(ax)): # delete useless axes
        fig.delaxes(ax[jj])
     
   
def  deperametrizeTasks(decompostionAgents : dict[int,AgentTaskDecomposition])-> dict[int,list[StlTask]]:
    """
        Simulate agents deparametrization of the the task. This happens as follows. For all the active parametric tasks, each agent can repolace the task with a non parameteric one given its solution for the active tasks.
        For the tasks that are passive and parameteric. I have to ask the respective neigbour for its solution such that I can then update the parameteric tasks with the given values found by the computational agent
    """
    agentTasksPair = {}
    
    
    for agent in decompostionAgents.values() :
        taskList = []
        task_containers = agent.taskConstainers # active task containers
        passivetask_containers = agent.passiveTaskConstainers # active task containers
        
        for task_container in task_containers :
            if task_container.task.is_parametric :
                center = agent.optimizer.value(task_container.task.center)
                scale  = agent.optimizer.value(task_container.task.scale)
                
                task_container.task.predicate.A
                # create a new task out of the parameteric one
                predicate = PolytopicPredicate(A     = task_container.task.predicate.A,
                                                     b     = scale*task_container.task.predicate.b,
                                                     center= center)
                task = StlTask(temporalOperator   = task_container.task.temporal_operator,
                                     timeinterval       = task_container.task.time_interval,
                                     predicate          = predicate,
                                     source             = task_container.task.sourceNode,
                                     target             = task_container.task.targetNode,
                                     timeOfSatisfaction = task_container.task.timeOfSatisfaction)
                taskList += [task]
                
                
            else : # no need for any parameter
                taskList += [task_container.task]
                        
        for task_container in passivetask_containers :
            if task_container.task.is_parametric :
                # ask the agent on the other side of the node for the solution
                if agent.agentID == task_container.task.sourceNode : # you are the source mnode ask the target node
        
                    for container in decompostionAgents[task_container.task.targetNode].taskConstainers :
                        if container.taskID == task_container.taskID :
                            center = decompostionAgents[task_container.task.targetNode].optimizer.value(container.task.center)
                            scale  = decompostionAgents[task_container.task.targetNode].optimizer.value(container.task.scale)
                            break # only one active task from the other agent will correspond to a passive task for this agent. Both tasks have the same ID
                    
                else : # you are the target node ask the source node
                    for container in decompostionAgents[task_container.task.sourceNode].taskConstainers :
                        if container.taskID == task_container.taskID :
                            center = decompostionAgents[task_container.task.sourceNode].optimizer.value(container.task.center)
                            scale  = decompostionAgents[task_container.task.sourceNode].optimizer.value(container.task.scale)
                            break
                    
                task_container.task.predicate.A
                # create a new task out of the parameteric one
                predicate = PolytopicPredicate(A     = task_container.task.predicate.A,
                                                     b     = scale*task_container.task.predicate.b,
                                                     center= center)
                task = StlTask(temporalOperator   = task_container.task.temporal_operator,
                                     timeinterval       = task_container.task.time_interval,
                                     predicate          = predicate,
                                     source             = task_container.task.sourceNode,
                                     target             = task_container.task.targetNode,
                                     timeOfSatisfaction = task_container.task.timeOfSatisfaction)
                
                taskList += [task]
            else : # no need for any parameter
                taskList += [task_container.task]
    
        agentTasksPair[agent.agentID] = {"tasks":taskList}
    
    
    return agentTasksPair
    
    
if __name__== "__main__" :
    pass
