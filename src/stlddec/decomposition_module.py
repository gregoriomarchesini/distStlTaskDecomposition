import numpy as np
import casadi as ca
import itertools
from .predicate_builder_module import * 
from .transport import Message,Publisher
from   typing import Self
import networkx as nx 
import matplotlib.pyplot as plt 
import logging
import polytope as poly
import copy


    
class EdgeDict(dict):
    """Helper dictionary class: takes undirected edges as input keys only"""
    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            raise TypeError("Keys must tuples of type (int,int)")
        if not len(key) == 2:
            raise TypeError("Keys must tuples of type (int,int)")
        return tuple(sorted(key))
    
    def __setitem__(self, key, value):
        normalized_key = self._normalize_key(key)
        super().__setitem__(normalized_key, value)

    def __getitem__(self, key):
        normalized_key = self._normalize_key(key)
        return super().__getitem__(normalized_key)

    def __delitem__(self, key):
        normalized_key = self._normalize_key(key)
        super().__delitem__(normalized_key)

    def __contains__(self, key):
        normalized_key = self._normalize_key(key)
        return super().__contains__(normalized_key)



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
        
        coupling_constraint_size = self.task.predicate.num_hyperplanes*self.task.predicate.num_verices # Size of the matrix M
        
        self._concensus_param_neighbors   = EdgeDict()
        self._lagrangian_param            = np.zeros((coupling_constraint_size,1)) # self lagrangian multiplies
        self._average_concensus_param     = np.zeros((coupling_constraint_size,1)) # average concensus parameter for the concesusn variable y
        
        
        # Initialize concensus parameters and lagrangian multiplies 
        for edge in neighbour_edges :
            self._concensus_param_neighbors[edge]  = np.zeros((coupling_constraint_size,1)) # contains the concensus variable that this task has for each neighbour (\lambda_{ij} in paper)
            
        # Define shared constraint (will be used to retrive the lagrangian multipliers for it after the solution is found)
        self._task_constraint             : ca.MX = None # this the constraint that each agent has set for the particular task. This is useful to geth the langrangian multiplier
        self._concensus_average_param     : ca.MX = None # average concensus parameter for the concesusn variable y
        

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
    def concensus_param_neighbors(self):
        return self._concensus_param_neighbors
    
    @property
    def concensus_average_param(self):
        return self._concensus_average_param
    
    @property
    def decompostion_path_length(self):
        return self._decomposition_path_length
    
    
    def set_optimization_variables(self,opti:ca.Opti):
        """Set the optimization variables for the task container"""
        self._center_var = opti.variable(self.task.predicate.state_space_dim,1)                            # center the parameteric formula
        self._scale_var  = opti.variable(1)                                                         # scale for the parametric formula
        self._eta_var     = ca.vertcat(self._center_var,self._scale_var)                                    # optimization variable
        
        coupling_constraint_size      = self.task.predicate.num_hyperplanes*self.task.predicate.num_verices # Size of the matrix M
        self._average_concensus_param = opti.parameter(coupling_constraint_size,1) # average concensus parameter for the concesusn variable y
        self._is_initialized          = True

    def set_task_constraint_expression(self, constraint : ca.MX) -> None :
        """Save the constraint for later retrieval of the lagrangian multipliers"""
        self._task_constraint   :ca.MX         = constraint
    
    
    def update_concensus_variable_of_neighbour_edge(self, edge: tuple[int,int], lagrangian_coefficients_neighbor: np.ndarray,learning_rate:float) -> None :
        """Update the concensus variable for a particular edge"""
        self._concensus_param[edge] -= (self._lagrangian_param - lagrangian_coefficients_neighbor)*learning_rate
    
    
    def compute_concensus_average_parameter(self, concensus_variable_from_neighbours:EdgeDict) -> np.ndarray :
        """computes sum_{j} \lambda_{ij} - \lambda_{ji}  
        
        Args: 
            concensus_variable_from_neighbours (EdgeDict) : concensus variable transmitted from the neighbours (\lambda_{ji} in paper)
        Returns:
            average (np.ndarray) : average of the concensus variable
        
        """
         
        average = 0
        for edge in concensus_variable_from_neighbours.keys() :
            average += concensus_variable_from_neighbours[edge] - self._concensus_param_neighbors[edge]
            
        return average
    

class AgentTaskDecomposition(Publisher) :
    """
       Agent Task Decomposition
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
    
    #todo: To be corrected still    
    def setUpOptimizer(self,numIterations :int) :
        """sets up the optimization problem for the given agent
        """    
        
        if not self._is_computing_agent :
            raise ValueError("Only computing agents can set up the optimization problem")
        
        
        if len(self._activetask_containers)!=0 and (not self._is_initialized_for_optimization) :# initialised only once when you have active tasks
            self._num_iterations     = int(numIterations) # number of iterations
            self._is_initialized_for_optimization  = True
    
            # here we need to set all the optimization variabled for your problem
            cost = 0
            for taskContainer in self._activetask_containers :
                
                task = taskContainer.task
                
                if task.isParametric :
                    task.setOptimizationVariables(self._optimizer) # create scale,center and concensus variables for each task
                    
                    # set the scale factor positive in constraint
                    self._optimizer.subject_to(task.scale >0)
                    self._optimizer.subject_to(task.scale <=1)
                    self._optimizer.set_initial(task.scale,0.2) # set an initial guess value
                    cost += -task.scale
            
            cost += 12*self._penalty # setting cost toward zero
       
        # set up private and shared constraints (also if you don't have constraints this part will be printed)
        privateConstraints = self._computeOverloadingConstraints()
        sharedConstraint   = self._compute_shared_constraints()
        print("-----------------------------------")
        print(f"{self._agent_id }")
        print(f"number of private constraints      : {len(privateConstraints)}")
        print(f"number of shared constraints       : {len(sharedConstraint)}")
        print(f"number of active  tasks            : {len(self._activetask_containers)}")
        print(f"number of passive tasks            : {len(self._passivetask_containers)}")
            
        if len(privateConstraints) != 0:
            self._optimizer.subject_to(privateConstraints) # it can be that no private constraints are present
        if len(sharedConstraint)!=0 :
            self._optimizer.subject_to(sharedConstraint)
        
        # set up cost and solver for the problem
        if  self._is_initialized_for_optimization :
            self._optimizer.minimize(cost)
            p_opts = dict(print_time=False, verbose=False)
            s_opts = dict(print_level=1)
    
            self._optimizer.solver("ipopt",p_opts,s_opts)
            self._cost = cost
        
    
    def _compute_shared_constraints(self) -> list[ca.Function]:
        """computes the shared inclusion constraint for the given agent. The shared constraints are the incluson of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints (list[ca.Function]): set of constraints
        """        
        
        constraints_list = []
        for container in self._task_containers :
            task                 = container.task                     # extract task
            num_computing_agents = container.decompostion_path_length -1  # Number of agents sharing the computation for this constraint
            
            # Compute constraints.
            M,Z  = get_M_and_Z_matrices_from_inclusion(P_including=task.parent_task, P_included=task) # get the matrices for the inclusion constraint
            Z    = Z/num_computing_agents # scale the Z matrix by the number of computing agents 
            constraint = (M@task.eta_var - Z)  - self._penalty + container.concensus_average_param <= 0 # set the constraint
            constraints_list = [constraint]
            
            container.set_task_constraint_expression(constraint=constraint)
        
        return constraints_list
        
     
            
    def _computeOverloadingConstraints(self) -> list[ca.Function]:
        """mutiple collaborative tasks

        Returns:
            constraints (list[ca.Function]): resturns constraints for overloading of the edge with multiple collaborative tasks
        """        
        
        constraints = []
    
        if len(self._activetask_containers) == 1 :
           return  [] # with one single task you don't need overloading constraints
        
        taskCombinations : list[tuple[TaskOptiContainer,TaskOptiContainer]] = list(itertools.combinations(self._activetask_containers,2))
        
        for taskContainerI,taskContainerJ in taskCombinations :
            if not (taskContainerI.task.time_interval  / taskContainerJ.task.time_interval).isEmpty() : # there is an intersection
                constraints += self._computeIntersectionConstraint(taskContainerI,taskContainerJ)
   
        return constraints    
    
    def _computeIntersectionConstraint(self,taskContaineri:TaskOptiContainer,taskContainerj:TaskOptiContainer) -> list[ca.Function] :
        """ creates the intersection constraionts for two tasks """
        
        taski : StlTask = taskContaineri.task
        taskj : StlTask = taskContainerj.task
        
        if (not taski.isParametric) and (not taskj.isParametric) : # if both the tasks are nonparameteric, there is no constraint to add
            return []
        
        # help the solver with the initial location of the point
        
        
        Ai = taski.predicate.A
        Aj = taskj.predicate.A
        bi = taski.predicate.b
        bj = taskj.predicate.b
        
        
        # centers
        if (taski.sourceNode== taskj.targetNode) and (taskj.sourceNode== taski.targetNode) : #the two have inverted source target pairs and then we have to invert one of the centers at least
            
            ci = taski.center 
            cj = -taskj.center
            
        elif (taski.sourceNode== taskj.sourceNode) and (taskj.targetNode== taski.targetNode) : # have the same traget source pairs
            ci = taski.center 
            cj = taskj.center
       
        else : # the two tasks do not pertain to the same edge. Hence there is no constraint to be included bceause the two tasks pertain to different edges
            return []
        
        
        # scale factors
        si = taski.scale
        sj = taskj.scale
        zeta = self._optimizer.variable(taski.stateSpaceDimension) # this is vector applied only to check an intersection (at every intersection you have a new define variable in the optimization)
        
        if not taski.isParametric :
            self._optimizer.set_initial(zeta,ci) # give an hint about the point for easier satisfaction of the intersaction constraint
        elif not taskj.isParametric :
            self._optimizer.set_initial(zeta,cj) # give an hint about the point for easier satisfaction of the intersaction constraint
            
        
        constraints = []
        # you enforce that there exists at least one point y that is contained in both the sets (t just says a certain degree of penetration)
        constraints += [Ai@(zeta-ci) - si*bi<=0 ] # t will be negative. The more negative and the more the two polygons will intersect
        constraints += [Aj@(zeta-cj) - sj*bj<=0 ]
        
        return constraints
   
    
    def unpackReponse(self, response : ConcensusVariableResposeMessage) -> None:
        """ here you upack the reponse for bith active and passive tasks"""
        
        # check in the active ones
        for taskContainer in self._activetask_containers :
            if (taskContainer.taskID == response.taskID) and (response.edge in taskContainer.edgeNeigboursConcensusVariable):
                
                taskContainer.updateNeigboursConcensusVariables(edgeNeigbour          = response.edge, 
                                                                lagrancianCoefficient = response.lagrangianCoefficient, 
                                                                concensusVariable     = response.concensusVariable) # update the neigbour concensus variable
                return 
        
        # check in the passive one
        for taskContainer in self._passivetask_containers :
            if (taskContainer.correspondsTo(response)):
                taskContainer.updateConcensusVariables(lagrancianCoefficient = response.lagrangianCoefficient, 
                                                       concensusVariable     = response.concensusVariable)
                return
                
        
    def replyConcensusRequest(self, request : ConcensusVariableRequestMessage) -> ConcensusVariableResposeMessage|None:
        """reply the concensus request"""
       
        # check first that you have the task among the active ones 
        for taskContainer in  self.task_containers :
            
            # you only reply if you have a task container with the same edge as the requested one and the same task ID
            if taskContainer.correspondsTo(request):
                lagrangianCoefficient = taskContainer.lagrangianCoefficient
                concensusVariable     = taskContainer.concensusVariable
                response  = ConcensusVariableResposeMessage(senderID = self._agent_id , 
                                                            taskID   = taskContainer.taskID,
                                                            edge     = request.edge,
                                                            lagrangianCoefficients = lagrangianCoefficient,
                                                            concensusVariable      = concensusVariable)
                return response # there is only one message per task, so you can break as soon as you found the one that you needed 
    
    def sendConcensusRequests(self) -> list[ConcensusVariableRequestMessage] :
        """Send a concensus request for the active tasks given to the agent"""
        requests : list[ConcensusVariableRequestMessage]  = []
        
        parametricActiveContainers  = [ taskContainer for taskContainer in self._activetask_containers if taskContainer.task.isParametric]
        parametricPassiveContainers = [ taskContainer for taskContainer in self._passivetask_containers if taskContainer.task.isParametric] 
        
        # ask for all the active tasks
        for taskContainer in parametricActiveContainers  :
            for edgeNeigbour in taskContainer.edgeNeigboursConcensusVariable.keys() : # compile a request for each neigbouring edge
                requests+= [ConcensusVariableRequestMessage(senderID=self.agentID,edge=edgeNeigbour,taskID=taskContainer.taskID)] 
        
        # for all the tasks that you have as active from one edge and active from the other edge, you do not need to send a 
        # request. Indeed, you can update the neigbours parameters for this tasks yourself if you wanted to. But this 
        # is actually not necessary to do and it can be skipped. The other active agents will ask you for updates if needed.
        # On the other hand you will ask for updates over the passive tasks that you don't have as active on another edge. So for those tasks
        # you will be a simple messanger basically.
        
        for PtaskContainer in parametricPassiveContainers :
            requests+= [ConcensusVariableRequestMessage(senderID=self.agentID,edge=PtaskContainer.task.sourceTarget,taskID=PtaskContainer.taskID)] # this is just asking your directed neigbours for updates
        
        return requests
    
    
    def updateY(self):
        
        learningRate = 0.9 * (1/self._current_iteration)**0.7
        for taskContainer in self._activetask_containers : # update concensus variables active task containers
            if taskContainer.task.isParametric :
                cAverage = taskContainer.computeLagrangianCoefficientsAverage()
                taskContainer.concensusVariable = taskContainer.concensusVariable - learningRate*(cAverage) # concensus update
                
    def solveLocalProblem(self,printResult:bool=False) -> None :
        
        
        # algorithm start assuming the the others solved their problem (because you all set the lagangina multipliers to zero)
        
        for taskContainer in self._activetask_containers :
            if taskContainer.task.isParametric :
                yAverage = taskContainer.computeConcensusAverage()
                self._optimizer.set_value(taskContainer.concensus_average_param,yAverage)
                
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
            print("penetration coefficient :",self._optimizer.debug.value(self._t))
            print("cost                    : ",self._optimizer.debug.value(self._cost))
            print("******************************************************************************")
            
            exit()
      
        
        if printResult :
            print("--------------------------------------------------------")
            print(f"Agent {self._agent_id } SOLUTION")
            print("--------------------------------------------------------")
            for taskContainer in self._activetask_containers :
                if taskContainer.task.isParametric :
                    print("Center and Scale")
                    print(self._optimizer.value(taskContainer.task.center))
                    print(self._optimizer.value(taskContainer.task.scale))
                    print("Lagrangian Concensus norm")
                    print(np.linalg.norm(taskContainer.computeLagrangianCoefficientsAverage()))
            
            print("penalty")
            print(self._optimizer.value(self._penalty))
            print("penetration conefficients")
            print(self._optimizer.value(self._t))
            print("--------------------------------------------------------")
        
        self._penalty_values.append(self._optimizer.value(self._penalty))
        self._cost_values.append(self._optimizer.value(self._cost))
        
        self._warm_start_solution = sol
        # update lagangian coefficients
        for taskContainer in self._activetask_containers :
            if taskContainer.task.isParametric :
                taskContainer.lagrangianCoefficient = self._optimizer.value(self._optimizer.dual(taskContainer.localConstraint))[:,np.newaxis]
        
        if not self._was_solved_already_once :
            self._was_solved_already_once = True
            self._optimizer.solver("sqpmethod",{"qpsol":"qpoases"}) 
            
        
        # save current iteration
        self._current_iteration += 1
        
    def getListOfActiveTasks(self) -> list[StlTask]:
        """
           Returns the list of active tasks for this agent. Indeed, the active tasks are all defined over one edge and they are the tasks that the agent needs to use to solve the parameters of the parameteric tasks. 
           Note that both parameteric and non parametric tasks are present here since both need to be used in the optimization. The returned list of tasks does not contain any parameteric task since the parameter were found as a resul of the 
           optimization.
        """

        # tasks that are not parameteric over the edge
        nonParametricTasks = [taskContainer.task for taskContainer in self._activetask_containers if not taskContainer.task.isParametric]
        
        solvedTasks = []
        if self._was_solved_already_once : # if there was an optimization
            for taskContainer in self._activetask_containers : # look in all the active tasks of the edge
                if taskContainer.task.isParametric : # check the ones that are parameherrioc and hence needs to be replaced by nonParametrci tasks after the optimzation
                    center = self._optimizer.value(taskContainer.task.center)
                    scale = self._optimizer.value(taskContainer.task.scale)
                    taskContainer.task.predicate.A
                    # create a new task out of the parameteric one
                    predicate = PolytopicPredicate(A     = taskContainer.task.predicate.A,
                                                         b     = scale*taskContainer.task.predicate.b,
                                                         center= center)
                    task = StlTask(temporalOperator   = taskContainer.task.temporal_operator,
                                         timeinterval       = taskContainer.task.time_interval,
                                         predicate          = predicate,
                                         source             = taskContainer.task.sourceNode,
                                         target             = taskContainer.task.targetNode,
                                         timeOfSatisfaction = taskContainer.task.timeOfSatisfaction
                                         )
                    
                    solvedTasks += [task]
                      
        tasksList = nonParametricTasks + solvedTasks    # return all the tasks defind over the edge. Non paramteric task is give here

        return tasksList
        







def concensusRound(decompositionAgents : dict[int,AgentTaskDecomposition],commGraph: nx.DiGraph,yOnly:bool=False) :
    """start an exchange of communication messages among the agents. This is a centralised simulation of what should happen in a decentralised way"""
    
    # we need this for loop to be run twice. One time for the active computing agents to share the information about the concensus parameters and one for the supporting agents to get the inforamtions to the two hop away agents
    for round in range(2) :
        for agentID,agent in decompositionAgents.items() :
            
            requests = agent.sendConcensusRequests() # compile requests
            neigbours = list(commGraph.neighbors(agentID))

            # simulate a communication round
            
            for request in requests :
                for neigbour in neigbours :
                    reply = decompositionAgents[neigbour].replyConcensusRequest(request=request)
                    if reply != None : # if there is a reply (you don't get one if you sedn a message tp the wrong neigbour)
                        print("---------request parameters--------")
                        print("asker",request.senderID)
                        print("edge  ",request.edge)
                        print("taskID ,",request.taskID)
                        print("---------reply parameters----------")
                        print("replier",reply.senderID)
                        print("edge  ",reply.edge)
                        print("taskID ,",request.taskID)
                        print("-----------------------------------")
                        agent.unpackReponse(reply) # get the reply and update the variables accordingly
                        break
   
            
def edgeSet(path:list[int],isCycle:bool=False) -> list[(int,int)] :
    
    if not isCycle :
      edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
    elif isCycle : # due to how networkx returns edges
      edges = [(path[i],path[i+1]) for i in range(-1,len(path)-1)]
        
    return edges


class GraphEdge( ) :
    """class GraphEdge 
    This class is useful to define attributes of the edges like STL predicate function, weights ...
    
    class attributes 
    centerVar (cvx.) : edge cvx variable
    nu      (cvx.Variable) : nu   cvx variable for hypercube stateSpaceDimensions
    predicateFunction(function) : function wrapper for predicate function
    """
    
    def __init__(self,source :int,target:int,isCommunicating :int= 0,weight:float=1) -> None:
     
      """ 
      Input
      ----------------------------------
      weight          (float)  : weight of the edge for shortest path algorithms
      isCommunicating (boolean) : 0 is a communicating edge, 1 is a communicating edge
      """
      
     
      if weight<=0 : # only accept positive weights
          raise("Edge weight must be positive")
      
      self._isCommunicating          = isCommunicating
      self._isInvolvedInOptimization = 0
      
      if not(self._isCommunicating) :
          self._weight = float("inf")
      else :
          self._weight = weight
          
      self._tasksList = []  
      self._task_containers = []
      
      if (not isinstance(source,int)) or (not isinstance(target,int)) :
          raise ValueError("Target source pairs must be integers")
      else :
          self._sourceNode = source
          self._targetNode = target
      
    @property
    def tasksList(self) :
        return self._tasksList
    @property
    def task_containers(self):
        return self._task_containers
    
    @property
    def isCommunicating(self) :
      return self._isCommunicating 
   
    @property
    def sourceNode(self) :
        return self._sourceNode
    @property
    def targetNode(self) :
        return self._targetNode
  
    @property
    def isInvolvedInOptimization(self) :
      return self._isInvolvedInOptimization
  
    @property
    def weight(self):
        return self._weight
    @property
    def hasSpecifications(self):
        return bool(len(self._tasksList)) 

    
    @weight.setter
    def weight(self,new_weight:float)-> None :
        if not isinstance(new_weight,float) :
            raise TypeError("Weight must be a float")
        elif new_weight<0 :
            raise ValueError("Weight must be positive")
        else :
            self._weight = new_weight
    
   
    def _addSingleTask(self,inputTask : StlTask|TaskOptiContainer) -> None :
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not (isinstance(inputTask,StlTask) or isinstance(inputTask,TaskOptiContainer)) :
            raise Exception("please enter a valid STL task object or a list of StlTask objects")
        else :
            if isinstance(inputTask,StlTask) :
                # set the source node pairs of this node
                if inputTask.hasUndefinedDirection :
                    inputTask.sourceTarget(source=self._sourceNode,target=self._targetNode)
                    self._task_containers.append(TaskOptiContainer(task=inputTask))
                    self._tasksList.append(inputTask) # adding a single task
                else :
                    if ((self._sourceNode,self._targetNode) != (inputTask.sourceNode,inputTask.targetNode)) and  ((self._targetNode,self._sourceNode) != (inputTask.sourceNode,inputTask.targetNode)):
                        raise Exception(f"Trying to add a task with defined source/target pair to an edge with different source/target pair. task has source/tarfet pair {inputTask.sourceTarget} and edge has {(self._sourceNode,self.targetNode)}")
                    else :
                        self._task_containers.append(TaskOptiContainer(task=inputTask))
                        self._tasksList.append(inputTask) # adding a single task
            
            
            elif isinstance(inputTask,TaskOptiContainer):
                # set the source node pairs of this node
                task = inputTask.task
                if task.hasUndefinedDirection :
                    task.sourceTarget(source=self._sourceNode,target=self._targetNode)
                    self._task_containers.append(inputTask) # you add directly the task container
                    self._tasksList.append(task) # adding a single task
                else :
                    if ((self._sourceNode,self._targetNode) != (task.sourceNode,task.targetNode)) and  ((self._targetNode,self._sourceNode) != (task.sourceNode,task.targetNode)):
                        raise Exception(f"Trying to add a task with defined source/target pair to an edge with different source/target pair. task has source/tarfet pair {task.sourceTarget} and edge has {(self._sourceNode,self.targetNode)}")
                    else :
                        self._task_containers.append(inputTask)
                        self._tasksList.append(task) # adding a single task
            
    
    def addTasks(self,tasks : StlTask|TaskOptiContainer|list[StlTask]|list[TaskOptiContainer]):
        if isinstance(tasks,list) : # list of tasks
            for  task in tasks :
                self._addSingleTask(task)
        else :# single task case
            self._addSingleTask(tasks)
    
 
    def flagOptimizationInvolvement(self) -> None :
        self._isInvolvedInOptimization = 1
        
    def cleanTasks(self)-> None :
        del self._task_containers
        del self._tasksList
        
        self._task_containers = []
        self._tasksList      = []
    
        
def computeWeights(source,taget,attributesDict) :
    """takes the edge object from the attributes and returns the wight stored in there"""
    return attributesDict["edgeObj"].weight    


def findEdge(edgeList : list[GraphEdge],edge:tuple[int,int])-> GraphEdge :
    
    ij = edge
    ji = (edge[1],edge[0])
    for edgeObj in edgeList :
        if (edgeObj.sourceNode,edgeObj.targetNode) == ij or (edgeObj.sourceNode,edgeObj.targetNode) == ji :
            return edgeObj
    
    edgesString = ""
    for edgeObj in edgeList :
        edgesString += f"{(edgeObj.sourceNode,edgeObj.targetNode)}"
    raise RuntimeError(f"The searched edge {edge} was not found. Available edges are : {edgesString}")


def runTaskDecomposition(edgeList: list[GraphEdge]) -> (nx.Graph,nx.Graph,nx.Graph,list[GraphEdge]):
    """Task decomposition pipeline"""
    
            
    # create communication graph
    commGraph = createCommunicationGraphFromEdges(edgeList=edgeList)
    # create task graph
    originalTaskGraph = createTaskGraphFromEdges(edgeList=edgeList)
    # create the agents for decomposition
    decompositionAgents : dict[int,AgentTaskDecomposition] = {}
    
    # crate a computing node for each node in the graph
    for node in commGraph.nodes :
        decompositionAgents[node] = AgentTaskDecomposition(agentID=node)
    
    pathList : list[int] =  []
    
    # for each each edge check if there is a decomposotion to be done
    for edgeObj in edgeList :
        if (not edgeObj.isCommunicating) and (edgeObj.hasSpecifications) : # decomposition needed
            # retrive all the tasks on the edge because such tasks will be decomposed
            tasksContainersToBeDecomposed: list[TaskOptiContainer] = edgeObj.task_containers
            path = nx.shortest_path(commGraph,source=edgeObj.sourceNode,target = edgeObj.targetNode)
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
  
            decompositionAgents[edgeObj.sourceNode].addActivetask_containers(edgeObj.task_containers)  # all the containers for the optimization (source node is the one makijng computations actively on this constraints)
            decompositionAgents[edgeObj.targetNode].addPassivetask_containers(edgeObj.task_containers) # all the containers for message resumbission (target node willl forward this information if necessary to other nodes)
               
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for agent in decompositionAgents.values() :
        agent.setUpOptimizer(numIterations=numOptimizationIterations)
    
    concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents)# share current value of private lagrangian coefficients and auxiliary y variable
    # find solution
    for jj in range(numOptimizationIterations) :
        for agentID,agent in decompositionAgents.items() :
            if agent.is_initialized_for_optimization : #only computing agents should make computations 
                agent.solveLocalProblem()
        concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents)
        # update y
        for agentID,agent in decompositionAgents.items() :
            if agent.is_initialized_for_optimization :
                agent.updateY() # update the value of y
        concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents) # concensus before leaving

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
    
    return commGraph,finalTaskGraph,originalTaskGraph


    
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
        
        for taskContainer in decomposedTasks :
            tasksAlongThePath = {}
            for agentID,agent in computingAgents.items() :
                agenttask_containers = agent.activeTaskConstainers # search among the containers you optimised for 
                for container in agenttask_containers : # check which task on the agent you have correspondance to
                    if taskContainer.taskID == container.taskID :
                        tasksAlongThePath[agent] = container # add the partial task now
            
            decompositionDictionay[taskContainer] = tasksAlongThePath  # now you have dictionary with parentTask and another dictionary will all the child tasks       
    
    
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
            yaverageSum += childTaskContainer.computeConcensusAverage()
            
            # print successive edge sets
            bprime = scaleSums*decomposedTaskContainer.task.predicate.b +  decomposedTaskContainer.task.predicate.A@centerSums
            poly.Polytope(A = decomposedTaskContainer.task.predicate.A,b= bprime).plot(ax=ax[counter],color="blue",alpha=0.3)
            ax[counter].quiver(previousCenter[0],previousCenter[1],centerSums[0]-previousCenter[0],centerSums[1]-previousCenter[1],angles='xy', scale_units='xy', scale=1)
            
            previousCenter += computingAgent.optimizer.value(childTaskContainer.task.center)
            
            
            
        print(f"Sum of scales         : {scaleSums}".ljust(30)      +     "(<= 1.)")
        print(f"Sum of Centers        : {centerSums}".ljust(30)     +     "(similar to original)")
        print(f"Concensus variables   : {np.sum(yaverageSum)}".ljust(30) + "(required 0.0)")
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
        activetask_containers = agent.activeTaskConstainers # active task containers
        passivetask_containers = agent.passiveTaskConstainers # active task containers
        
        for taskContainer in activetask_containers :
            if taskContainer.task.isParametric :
                center = agent.optimizer.value(taskContainer.task.center)
                scale  = agent.optimizer.value(taskContainer.task.scale)
                
                taskContainer.task.predicate.A
                # create a new task out of the parameteric one
                predicate = PolytopicPredicate(A     = taskContainer.task.predicate.A,
                                                     b     = scale*taskContainer.task.predicate.b,
                                                     center= center)
                task = StlTask(temporalOperator   = taskContainer.task.temporal_operator,
                                     timeinterval       = taskContainer.task.time_interval,
                                     predicate          = predicate,
                                     source             = taskContainer.task.sourceNode,
                                     target             = taskContainer.task.targetNode,
                                     timeOfSatisfaction = taskContainer.task.timeOfSatisfaction)
                taskList += [task]
                
                
            else : # no need for any parameter
                taskList += [taskContainer.task]
                        
        for taskContainer in passivetask_containers :
            if taskContainer.task.isParametric :
                # ask the agent on the other side of the node for the solution
                if agent.agentID == taskContainer.task.sourceNode : # you are the source mnode ask the target node
        
                    for container in decompostionAgents[taskContainer.task.targetNode].activeTaskConstainers :
                        if container.taskID == taskContainer.taskID :
                            center = decompostionAgents[taskContainer.task.targetNode].optimizer.value(container.task.center)
                            scale  = decompostionAgents[taskContainer.task.targetNode].optimizer.value(container.task.scale)
                            break # only one active task from the other agent will correspond to a passive task for this agent. Both tasks have the same ID
                    
                else : # you are the target node ask the source node
                    for container in decompostionAgents[taskContainer.task.sourceNode].activeTaskConstainers :
                        if container.taskID == taskContainer.taskID :
                            center = decompostionAgents[taskContainer.task.sourceNode].optimizer.value(container.task.center)
                            scale  = decompostionAgents[taskContainer.task.sourceNode].optimizer.value(container.task.scale)
                            break
                    
                taskContainer.task.predicate.A
                # create a new task out of the parameteric one
                predicate = PolytopicPredicate(A     = taskContainer.task.predicate.A,
                                                     b     = scale*taskContainer.task.predicate.b,
                                                     center= center)
                task = StlTask(temporalOperator   = taskContainer.task.temporal_operator,
                                     timeinterval       = taskContainer.task.time_interval,
                                     predicate          = predicate,
                                     source             = taskContainer.task.sourceNode,
                                     target             = taskContainer.task.targetNode,
                                     timeOfSatisfaction = taskContainer.task.timeOfSatisfaction)
                
                taskList += [task]
            else : # no need for any parameter
                taskList += [taskContainer.task]
    
        agentTasksPair[agent.agentID] = {"tasks":taskList}
    
    
    return agentTasksPair
    
    
      
def createCommunicationGraphFromEdges(edgeList : list[GraphEdge]) :
    """ builts a undirected graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    commGraph = nx.Graph()
    for edge in edgeList :
        if edge.isCommunicating : 
            commGraph.add_edge(edge.sourceNode,edge.targetNode)
    return commGraph

def createTaskGraphFromEdges(edgeList : list[GraphEdge]) :
    """ builts a undirected graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    taskGraph = nx.Graph()
    for edge in edgeList :
        if edge.hasSpecifications : 
            taskGraph.add_edge(edge.sourceNode,edge.targetNode)
    return taskGraph
    
    
    

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
