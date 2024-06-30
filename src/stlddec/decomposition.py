import numpy as np
import casadi as ca
from itertools import chain,combinations
from .stl_task import * 
from .transport import Publisher
from .graphs import *
from   typing import Self
import networkx as nx 
import matplotlib.pyplot as plt 
import logging
import polytope as poly
import copy



def edgeSet(path:list[int],isCycle:bool=False) -> list[(int,int)] :
    
    if not isCycle :
      edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
    elif isCycle : # due to how networkx returns edges
      edges = [(path[i],path[i+1]) for i in range(-1,len(path)-1)]
        
    return edges

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))




class TaskOptiContainer :
    """Storage of variables for each task to be decomposed (Only collaborative tasks)"""
    
    def __init__(self, 
                 task           : StlTask,
                 neighbour_edges: list[tuple[UniqueIdentifier,UniqueIdentifier]],
                 decompositon_path_length : int) -> None:
        
        """
        Args:
            task (StlTask): Task to be decomposed (Only collaborative tasks are considered)
            opti (ca.Opti): Instance of the optimizer applied for the task decomposition (this will in turn be specific of an agent)
            neighbours (list[UniqueIdentifier]) : contains the neighbours of the task decomposition
        """    
         
        self._task = task
        
        if isinstance(self.task.predicate,CollaborativePredicate):
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
            
        if self._task.is_parametric and (task.parent_task_id == None) :
            return ValueError("The task is parametric but the parent task ID is not set. Please set the parent task ID as it is required to run the decomposition")
        
        
        self._is_initialized = False
        # Create variables for optimization.
        self._center_var     = None 
        self._scale_var      = None 
        self._eta_var        = None 
                                                          
        self._decomposition_path_length = decompositon_path_length # length of the path for the decomposition
        
        coupling_constraint_size = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        
        self._consensus_param_neighbors_from_neighbors   = UndirectedEdgeMapping[np.ndarray]() # The value of the consensus parameters as computed and transmitted from the neighbors to this agent
        self._consensus_param_neighbors_from_self        = UndirectedEdgeMapping[np.ndarray]() # The value of the consensus parameters as computed from the agent itself.
        self._lagrangian_param_neighbors_from_neighbors  = UndirectedEdgeMapping[np.ndarray]() # value of the lagrangian optimal coefficients as computed from the agent.
        
        
        self._lagrangian_param_neighbors_from_self      = np.zeros((coupling_constraint_size,1)) # value of the lagrangian optimal coefficients as computed from the agent.
        
        
        self._average_consensus_param     = np.zeros((coupling_constraint_size,1)) # Average consensus parameter.
        self._neighbor_edges              = neighbour_edges # list of neighbour edges
        
        # Initialize consensus parameters and lagrangian multiplies 
        for edge in neighbour_edges :
            self._consensus_param_neighbors_from_neighbors[edge]   = np.zeros((coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ij} in paper)
            self._consensus_param_neighbors_from_self[edge]        = np.zeros((coupling_constraint_size,1)) # contains the consensus variable that this task has for each neighbour (\lambda_{ji} in paper)
            self._lagrangian_param_neighbors_from_neighbors[edge]  = np.zeros((coupling_constraint_size,1)) # lagrangian_coefficient from neighbur 
            
            
        # Define shared constraint (will be used to retrive the lagrangian multipliers for it after the solution is found)
        self._task_constraint             : ca.MX = None # this the constraint that each agent has set for the particular task. This is useful to geth the langrangian multiplier
        self._consensus_average_param     : ca.MX = None # average consensus parameter for the concesusn variable y
        

    @property
    def center_var(self):
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
        return self._center_var
    @property
    def scale_var(self):
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
        return self._scale_var
    @property
    def eta_var(self):
        if not self._is_initialized :
            raise ValueError("The optimization variables have not been initialized yet since no optimizer was provider for the container")
        return self._eta_var
    @property
    def task(self):
        return self._task
    
    @property
    def local_constraint(self):
        if self._task_constraint != None :
            return self._task_constraint
        else :
            raise RuntimeError("local constraint was not set. call the method ""saveConstraint"" in order to set a local constraint")
    @property
    def consensus_param_neighbors_from_neighbors(self):
        return self._consensus_param_neighbors_from_neighbors
    
    @property
    def consensus_param_neighbors_from_self(self):
        return self._consensus_param_neighbors_from_self
    
    @property
    def consensus_average_param(self):
        return self._consensus_average_param
    
    @property
    def decomposition_path_length(self):
        return self._decomposition_path_length
    
    
    def set_optimization_variables(self,opti:ca.Opti):
        """Set the optimization variables for the task container"""
        self._center_var = opti.variable(self.task.predicate.state_space_dim,1)                            # center the parameteric formula
        self._scale_var  = opti.variable(1)                                                         # scale for the parametric formula
        self._eta_var     = ca.vertcat(self._center_var,self._scale_var)                                    # optimization variable
        
        coupling_constraint_size      = self.task.predicate.num_hyperplanes*self.task.predicate.num_vertices # Size of the matrix M
        self._average_consensus_param = opti.parameter(coupling_constraint_size,1) # average consensus parameter for the concesusn variable y
        self._is_initialized          = True

    def set_task_constraint_expression(self, constraint : ca.MX) -> None :
        """Save the constraint for later retrieval of the lagrangian multipliers"""
        self._task_constraint   :ca.MX         = constraint
    
    def update_consensus_variable_from_neighbor(self,edge:tuple[int,int],consensus_variable:np.ndarray) -> None :
         self._consensus_param_neighbors_from_neighbors[edge] = consensus_variable
        
    def update_consensus_variable_from_self(self,edge:tuple[int,int],consensus_variable:np.ndarray,learning_rate:float) -> None :
        self._consensus_param_neighbors_from_self[edge] -= (self._lagrangian_param_neighbors_from_self[edge] - self._lagrangian_param_neighbors_from_neighbors[edge])*learning_rate
         
    def update_lagrangian_variable_from_neighbor(self,edge:tuple[int,int],lagrangian_variable:np.ndarray) -> None :
         self._lagrangian_param_neighbors_from_neighbors[edge] = lagrangian_variable
    
    def update_lagrangian_variable_from_self(self,lagrangian_variable:np.ndarray) -> None :
          self._lagrangian_param_neighbors_from_self= lagrangian_variable
    
    def compute_consensus_average_parameter(self,) -> np.ndarray :
        """computes sum_{j} \lambda_{ij} - \lambda_{ji}  
        
        Args: 
            consensus_variable_from_neighbours (EdgeDict) : consensus variable transmitted from the neighbours (\lambda_{ji} in paper)
        Returns:
            average (np.ndarray) : average of the consensus variable
        
        """
        average = 0
        for edge in self._neighbor_edges:
            average += self._consensus_param_neighbors_from_self[edge] - self._consensus_param_neighbors_from_neighbors[edge]
            
        return average



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
    def __init__(self,agent_id : UniqueIdentifier,is_computing_agent:bool) -> None:
        """
        Args:
            agentID (int): Id of the agent
        """        
        
        super().__init__()
        
        self._optimizer       : ca.Opti          = ca.Opti() # optimizer for the single agent.
        self._tasks           : list[StlTask]    = []        # list of tasks that the agent has to include in its optiization prgram.
        self._agent_id        : int              =  agent_id # agent ID.
        self._task_containers : list[TaskOptiContainer] = [] # list of task containers.
        self._is_computing_agent : bool = is_computing_agent # flag to indicate if the agent is a computing agent or not
       
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
        
    @property
    def agent_id(self):
        return self._agent_id 

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
    def is_computing_agent(self):
        return self._is_computing_agent
    
        
    def add_task(self,task:StlTask,neighbour_edges : list[tuple[UniqueIdentifier]]) :
        """Adds a single task to the computing node for the decomposition

        Args:
            task (StlTask): Stl task assigned to this computing agent to decompose
            neighbour_edges (list[tuple[UniqueIdentifier]]): Neighbouring edges to the task
        """
        
        if not isinstance(task.predicate.predicate,CollaborativePredicate) :
            raise ValueError("Only collaborative tasks are allowed to be decomposed")
        
        self._tasks.append(task)
        
        if not all([len(edge)==2 for edge in neighbour_edges]) :
            raise ValueError("The edges must be a list of tuples (int,int) of size 2")
        
        task_container = TaskOptiContainer(task=task,opti=self._optimizer,neighbour_edges=neighbour_edges)
        self._task_containers.append(task_container)
        
        # create the optimization variables for the solver
        if self._is_computing_agent :
            task_container.set_optimization_variables(self._optimizer)
            
            
    def _compute_shared_constraints(self) -> list[ca.MX]:
        """computes the shared inclusion constraint for the given agent. The shared constraints are the incluson of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints (list[ca.Function]): set of constraints
        """        
        
        constraints_list = []
        for container in self._task_containers :
            task                 = container.task                     # extract task
            num_computing_agents = container.decomposition_path_length -1  # Number of agents sharing the computation for this constraint
            
            # Compute constraints.
            M,Z  = get_M_and_Z_matrices_from_inclusion(P_including=task.parent_task, P_included=task) # get the matrices for the inclusion constraint
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
        eventually_task_containers = [task_container for task_container in self._task_containers if isinstance(task_container.task.temporal_operator,EventuallyOperator)]
        
        
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
        # Find all always tasks that intersect a single eventually task.
        intersection_sets = {}
        for eventually_container in eventually_task_containers :
            intersecting_always_tasks = []
            for always_container in always_task_containers :
                if  not (eventually_container.task.temporal_operator.time_interval / always_container.task.temporal_operator.time_interval ).is_empty() :
                    intersecting_always_tasks == [always_container] 
            
            intersection_sets[eventually_container] = intersecting_always_tasks
            
            
        # For each eventually task compute the power set of always tasks that intersect it.
        power_sets = { eventually_container : list(powerset(intersecting_always_tasks)) for eventually_container,intersecting_always_tasks in intersection_sets.items() } 
        
        for eventually_container,always_container_set in power_sets.items() :
            # Compute the intersection of all tasks in this subset.
            intersection = TimeInterval(a = float("-inf"),b = float("inf"))
            
            for always_task in always_container_set :
                intersection = intersection / always_task.task.temporal_operator.time_interval
            
            # At last check intersection with the eventually task.
            intersection = intersection / eventually_container.task.temporal_operator.time_interval
            
            # If there is a nonzero intersection then add a common intersection constraint.
            if not intersection.is_empty() :
                zeta = self._optimizer.variable(eventually_container.task.state_space_dimension)
                
                # Always formulas intersection.
                for always_container in always_container_set :
                    constraints += [get_inclusion_contraint(zeta = zeta, task_container = always_container)]
                
                
                constraints += [ get_inclusion_contraint(zeta = zeta, task_container = always_container) ]
                self._zeta_variables += [zeta]
        
        return constraints    
  
    
    def setup_optimizer(self,num_iterations :int) :
        """
        setup the optimization problem for the given agent
        """    
        
        if not self._is_computing_agent :
            raise ValueError(f"Only computing agents can set up the optimization problem including their constraints and cost function. The current agent with ID {self._agent_id} is not a computing agent.")
        
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
        print(f"{self._agent_id }")
        print(f"Number of overloading_constraints  : {len(overloading_constraints)}")
        print(f"Number of shared constraints       : {len(shared_path_constraint )}")
        print(f"Number of variables                : {len(self.optimizer.nx)}")
        
        # set up cost and solver for the problem
        self._optimizer.minimize(cost)
        p_opts = dict(print_time=False, verbose=False)
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
            logging.error(f'The optimization for agent {self._agent_id } failed with output: %s', e)
            print("******************************************************************************")
            print("The latest value for the variables was the following : ")
            print("penalty                 :",self._optimizer.debug.value(self._penalty))
            print("cost                    :",self._optimizer.debug.value(self._cost))
            print("******************************************************************************")
            
            exit()
      
        
        if print_result :
            print("--------------------------------------------------------")
            print(f"Agent {self._agent_id } SOLUTION")
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
    
    
    def update_consensus_variable_from_edge_neighbour_callback(self,consensus_variable:np.ndarray,edge:tuple[UniqueIdentifier,UniqueIdentifier],parent_task_id:int) -> None :
        """Update the task of the agent"""
        for task_container in self._task_containers :
            if task_container.task.parent_task_id == parent_task_id : # you have a task connected to the same parent id
                task_container.update_consensus_variable_from_neighbor(edge=edge,consensus_variable= consensus_variable)
                break
        
    def update_lagrangian_variable_from_edge_neighbour_callback(self,lagrangian_variable:np.ndarray,edge:tuple[UniqueIdentifier,UniqueIdentifier],parent_task_id:int) -> None :
        """Update the task of the agent"""
        for task_container in self._task_containers :
            if task_container.task.parent_task_id == parent_task_id :
                task_container.update_lagrangian_variable_from_neighbor(edge=edge,lagrangian_variable= lagrangian_variable)
        


def run_task_decomposition(edgeList: list[GraphEdge]) -> (nx.Graph,nx.Graph,nx.Graph,list[GraphEdge]):
    """Task decomposition pipeline"""
            
    # create communication graph
    comm_graph         = create_communication_graph_from_edges(edgeList=edgeList)
    # create task graph
    original_task_graph = create_task_graph_from_edges(edgeList=edgeList)
    # create the agents for decomposition
    decompositionAgents : dict[int,EdgeComputingAgent] = {}
    
    # crate a computing node for each node in the graph
    for node in comm_graph.nodes :
        decompositionAgents[node] = AgentTaskDecomposition(agentID=node)
    
    pathList : list[int] =  []
    
    # for each each edge check if there is a decomposotion to be done
    for edgeObj in edgeList :
        if (not edgeObj.isCommunicating) and (edgeObj.hasSpecifications) : # decomposition needed
            # retrive all the tasks on the edge because such tasks will be decomposed
            tasksContainersToBeDecomposed: list[TaskOptiContainer] = edgeObj.task_containers
            path = nx.shortest_path(comm_graph,source=edgeObj.sourceNode,target = edgeObj.targetNode)
            edgesThroughPath = edgeSet(path=path) # find edges along the path
            
            # update path list
            pathList.append(path)
            
            for taskContanier in tasksContainersToBeDecomposed : # add a new set of tasks for each edge along the path of the task to be decomposed
                
                task = taskContanier.task
                
                if taskContanier.task.sourceTarget != (path[0],path[-1]) : #case the direction is incorrect
                    taskContanier.task.flip() # flip the task such that all tasks have the same diretion of the node
                
             
                for sourceNode,targetNode in  edgesThroughPath : # the edges here are directed
                    # find subtask edge in the decomposition
                    subtaskEdge = findEdge(edgeList = edgeList,edge = (sourceNode,targetNode))
                    # create new subtask
                    subtask = create_parameteric_task_from(task = task,source = sourceNode,target = targetNode) # creates a parameteric copy of the original task
                    subTaskContainer = TaskOptiContainer(task = subtask,taskID = taskContanier.taskID ,path = path)
                    subtaskEdge.addTasks(subTaskContainer) 
    
        
    optimizationEdges = [] 
    for path in  pathList :
        optimizationEdges += edgeSet(path)
    
    optimizationEdges = list(set(optimizationEdges)) # unique set of edge unsed for the optimization
    numOptimizationIterations = 1000
    
    
    for edgeObj in edgeList :
   
        if edgeObj.isCommunicating and edgeObj.hasSpecifications : # for all the cummincating agent start putting the tasks over the nodes
  
            decompositionAgents[edgeObj.sourceNode].addtask_containers(edgeObj.task_containers)  # all the containers for the optimization (source node is the one makijng computations actively on this constraints)
            decompositionAgents[edgeObj.targetNode].addPassivetask_containers(edgeObj.task_containers) # all the containers for message resumbission (target node willl forward this information if necessary to other nodes)
               
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for agent in decompositionAgents.values() :
        agent.setUpOptimizer(numIterations=numOptimizationIterations)
    
    consensusRound(commGraph=comm_graph,decompositionAgents=decompositionAgents)# share current value of private lagrangian coefficients and auxiliary y variable
    # find solution
    for jj in range(numOptimizationIterations) :
        for agentID,agent in decompositionAgents.items() :
            if agent.is_initialized_for_optimization : #only computing agents should make computations 
                agent.solveLocalProblem()
        consensusRound(commGraph=comm_graph,decompositionAgents=decompositionAgents)
        # update y
        for agentID,agent in decompositionAgents.items() :
            if agent.is_initialized_for_optimization :
                agent.updateY() # update the value of y
        consensusRound(commGraph=comm_graph,decompositionAgents=decompositionAgents) # consensus before leaving

    fig,(ax1,ax2) = plt.subplots(1,2)
    for agentID,agent in decompositionAgents.items() :
        if agent.is_initialized_for_optimization :
            ax1.plot(range(numOptimizationIterations),agent._penaltyValues,label=f"Agent :{agentID}")
            ax1.set_ylabel("penalties")
            
            ax2.plot(range(numOptimizationIterations),agent._cost_values,label=f"Agent :{agentID}")
            ax2.set_ylabel("cost")
    
    ax1.legend()
    ax2.legend()
    printDecompositionResult(decompositionPaths =pathList, decompositionAgents =decompositionAgents, edgeObjectsLists = edgeList)
    
    # now that the tasks have been decomposed it is time to clean the old tasks and construct the new taskGraph
    for edgeObj in edgeList :
        if (not edgeObj.isCommunicating) and (edgeObj.hasSpecifications) :
            edgeObj.cleanTasks()     # after you have decomposed the tasks you can just clean them
    
    finalTaskGraph : nx.Graph = createTaskGraphFromEdges(edgeList=edgeList)
    
    nodesAttributes = deperametrizeTasks(decompostionAgents=decompositionAgents)
 
    # set the attributes
    nx.set_node_attributes(finalTaskGraph,nodesAttributes)
    
    return comm_graph,finalTaskGraph,original_task_graph


    
def printDecompositionResult(decompositionPaths : list[list[int]], decompositionAgents : dict[int,AgentTaskDecomposition], edgeObjectsLists : list[GraphEdge]) :
   
   
    # now we need to extract informations for each task 
    decompositionDictionay : dict[TaskOptiContainer,dict[tuple[int,int],TaskOptiContainer]]= {}
    
    # select the agents that have actually made computations 
    computingAgents = {agentID:agent for agentID,agent in decompositionAgents.items() if agent.is_initialized_for_optimization}
    
    
    for path in decompositionPaths :
        # now take initial and final index since this is the non communicating edge 
        (i,j) = path[0],path[-1]
        edgeObj = findEdge(edgeList = edgeObjectsLists, edge =(i,j)) # get the corresponding edge
        decomposedTasks : list[TaskOptiContainer] = edgeObj.task_containers                                # get all the tasks defined on this edge
        
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
        print(f"TASK ID           : {decomposedTaskContainer.taskID}")
        print(f"Temporal operator : {decomposedTaskContainer.task.temporal_operator}")
        print(f"Original Edge     : {decomposedTaskContainer.task.sourceTarget}")
        print(f"Original Center   : {decomposedTaskContainer.task.center}")
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
    
    
      

########################################################################################################################### 
# Visualization
###########################################################################################################################



def visualizeGraphs(communicationGraph:nx.Graph, initialTaskGraph:nx.Graph, finalTaskGraph:nx.Graph) :
    
    nodes = communicationGraph.nodes(data=True)
    xx = [node[1]["pos"][0] for node in nodes]
    yy = [node[1]["pos"][1] for node in nodes]
    xxmin,xxmax= min(xx)*1.6,max(xx)*1.6
    yymin,yymax = min(yy)*1.6,max(yy)*1.6

    nodes = communicationGraph.nodes(data=True)
    
    # define figure object
    fig, ax = plt.subplots() 
    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])


    # Define graphs
    fig, ax = plt.subplots(1,3) 

    # Communicatino Graph
    ax[0].set_xlim([xxmin,xxmax])
    ax[0].set_ylim([yymin,yymax])
    # # Drawing of the network
    edgeLabels = { (i,j):"link" for  i,j in communicationGraph.edges}    
    nx.draw_networkx(communicationGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[0])

    nx.draw_networkx_edge_labels(
        communicationGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[0]
    )

    ax[0].set_title("communication graph")


    # final Task Graph
    ax[1].set_xlim([-20,20])
    ax[1].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in finalTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(finalTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[1])

    nx.draw_networkx_edge_labels(
        finalTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[1]
    )

    ax[1].set_title("final Task Graph")
    
    ax[1].set_xlim([xxmin,xxmax])
    ax[1].set_ylim([yymin,yymax])
    
    
    # Initial Task Graph
    ax[2].set_xlim([-20,20])
    ax[2].set_ylim([-20,20])
    # # Drawing of the network
    edgeLabels = { (i,j):"Task" for  i,j,attr in initialTaskGraph.edges(data=True) if attr["edgeObj"].hasSpecifications}    
    taskPlot = nx.draw_networkx(initialTaskGraph,{node:nodeDict["pos"] for node,nodeDict in nodes},ax=ax[2])

    nx.draw_networkx_edge_labels(
        initialTaskGraph,
        {node:nodeDict["pos"] for node,nodeDict in nodes},
        edge_labels = edgeLabels,
        font_color='black',
        ax=ax[2]
    )

    ax[2].set_title("Initial Task Graph")
    ax[2].set_xlim([xxmin,xxmax])
    ax[2].set_ylim([yymin,yymax])







if __name__== "__main__" :
    pass
