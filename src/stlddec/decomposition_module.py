import numpy as np
import casadi as ca
import itertools
from .predicate_builder_module import * 
from   typing import Self
import networkx as nx 
import matplotlib.pyplot as plt 
import uuid
import logging
import polytope as poly
import copy


class edgeMapping(dict):
    """Support class for edge property storage"""

    def __setitem__(self, edge : [int,int], item):
        
        if not isinstance(edge,tuple) :
            raise ValueError(f"{self.__class__.__name__} accepts only tuple of integers as input. Given input was {edge}")
        elif isinstance(edge,tuple) and (len(edge)>2):
            raise ValueError(f"{self.__class__.__name__} accepts only tuple of integers of length 2 as input. Given input was {edge}")
        elif isinstance(edge,tuple) and (not isinstance(edge[0],int) or not isinstance(edge[1],int)):
            raise ValueError(f"{self.__class__.__name__} accepts only tuple of integers of length 2 as input. Given input was {edge}")
        elif edge in self : # don't add the key if the oppostite edge is already present
            for key in self.__dict__.keys() :
                if edge == key or ((edge[1],edge[0])== key) :
    
                    self.__dict__[key] = item
        else: # case in which the edge was not already present there            
            self.__dict__[edge] = item

    def __getitem__(self, edge):
        
        # try both the edges
        try :
            return self.__dict__[edge]
        except :
            try :
                self.__dict__[(edge[1],edge[0])]
            except :
                raise KeyError(f"The requested edge {edge} is not available in thsi edgeMapping. Availbale edges are {list(self.__dict__.keys())}")
            
    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, edge):
        del self.__dict__[edge]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, edge):
        for edgeCheck in self.__dict__.keys() :
            if edge == edgeCheck  or (edge[1],edge[0])== edgeCheck:
                return True
        return False
        

    def __iter__(self):
        return iter(self.__dict__)



class TaskContainer() :
    """Support class for task storage. Each task container contains the lagrangian coefficients and concensus variables for each task"""
    def __init__(self, task : StlTask , path : list[int] = [],taskID: int|None = None) -> None:
        """
        support class for distributed concensus optimization

        Args:
            path (list[int])    : path over which the task was decomposed
            task (StlTask): task assigned (contains information about the edge)
            k    (int, optional): index identifier of thi task (if multiple tasks are decompoised along the same path, then you have different index k for each)

        """    
         
        if not isinstance(path,list) :
            raise ValueError(" ""path"" must be a list of integers")
        else :
            self._path = path
        
        if not isinstance(task,StlTask) :
            raise ValueError(f"Input must be a valid instance of {StlTask.__class__.__name__}")
            
        else :
            self._task = task
        
        if taskID == None :
            self._taskID = uuid.uuid1().int
        else :
            self._taskID = taskID
        
        if self._task.isParametric and self._taskID == None :
            raise ValueError("for a parametric task you eed to assigne the corresponding parent task unique ID using the method id()")
        
        if (self._task.isParametric) and (len(self._path)==0) :
            raise ValueError("The provided task is parameteric but no path for the decomposition was provided. Please provide a decomposition path")
        
        # define concensus variables
        yDim   = task.predicate.numHyperplanes*task.predicate.numVerices  # 1 for the constraint (sum scales -1 <=0)
        self._concensusVariable       = np.zeros((yDim,1)) # concensus variable
        self._lagrangianCoefficient   = np.zeros((yDim,1)) # lagrangian coefficient
        
        # define shared constraint (will be used to retrive the lagrangian multipliers for it after the solution is found)
        self._localConstraint : ca.MX = None # this the constraint that each agent has set for the particular task. This is useful to geth the langrangian multiplier
        self._concensusVariableAveragePar     : ca.MX = None # average concensus parameter for the concesusn variable y
        
        # storage dictionaty for lagrangian coefficients anf concensus variables of the neigbours
        self._edgeNeigboursLangrangianCoefficients  = edgeMapping() # lagarngina coefficients
        self._edgeNeigboursConcensusVariable        = edgeMapping() # concensus variable for neigbours 
        self._isIntialised                          = False
        self.initialiseConcensusVariablesForNeigbours() # for each taskContainer you need to initialise the concensus variables
  
  
    @property
    def path(self):
        return self._path
    @property
    def taskID(self):
        return self._taskID
    @property
    def task(self):
        return self._task
    @property
    def localConstraint(self):
        if self._localConstraint != None :
            return self._localConstraint
        else :
            raise RuntimeError("local constraint was not set. call the method ""saveConstraint"" in order to set a local constraint")
    @property
    def concensusVariableAveragePar(self):
        return self._concensusVariableAveragePar
    
    @property 
    def yAveragePar(self):
        return self._concensusVariableAveragePar
    
    @property
    def concensusVariable(self):
        return self._concensusVariable
    
    @concensusVariable.setter
    def concensusVariable(self,value) :
        self._concensusVariable = value
    
    @property
    def lagrangianCoefficient(self):
        return self._lagrangianCoefficient
    
    @lagrangianCoefficient.setter 
    def lagrangianCoefficient(self,value):
        self._lagrangianCoefficient = value
        
    @property
    def edgeNeigboursLangrangianCoefficients(self):
        return self._edgeNeigboursLangrangianCoefficients
    
    @property
    def edgeNeigboursConcensusVariable(self):
        return self._edgeNeigboursConcensusVariable
    
    
    def saveConstraintParameters(self, constraint : ca.MX, concensusVariable: np.ndarray, concensusVariableAveragePar : ca.MX) -> None :
        """Save the constraint for later retrival of the lagarngian multipliers"""
        
        self._localConstraint   :ca.MX                 = constraint
        self._concensusVariable :np.ndarray            = concensusVariable            # current value of the concensus variable
        self._concensusVariableAveragePar :ca.MX       = concensusVariableAveragePar  # parameter of the concensus average
    
    
    def initialiseConcensusVariablesForNeigbours(self) -> None:
        """intialises container for a given agentID along the path. This funciton will set the neigbours for the given agentID as considered along the the decomposition path"""
        if not self._isIntialised :
            self._isIntialised = True
        
        if len(self._path)==0 : #means that there is not a path and hence the task does not need to be decomposed
            return 
        
        yDim   = self.task.predicate.numHyperplanes*self.task.predicate.numVerices  # +1 for the scale factor constrains sum(scales) = 1
        directedEdges = edgeSet(self.path)
        
        for jj,edge in enumerate(directedEdges) :
            if (self.task.sourceTarget  == edge) or (self.task.sourceTarget  == (edge[1],edge[0])) :
            
                if jj == 0: # if you are the first agent in the path, then your only neigbour is the one in front of you
                    self._edgeNeigboursLangrangianCoefficients.update({directedEdges[jj+1]:np.random.random((yDim,1))}) # only one neigbour
                    self._edgeNeigboursConcensusVariable.update({directedEdges[jj+1]:np.random.random((yDim,1))}) # only one neigbour 
                elif jj == (len(directedEdges)-1) : # if you are the last edge
                    self._edgeNeigboursLangrangianCoefficients.update({directedEdges[jj-1]:np.random.random((yDim,1))}) # only one neigbour
                    self._edgeNeigboursConcensusVariable.update({directedEdges[jj-1]:np.random.random((yDim,1))}) # only one neigbour 
                else : # in all the other cases your negbours are infront and on the back of you
                    self._edgeNeigboursLangrangianCoefficients.update({directedEdges[jj+1]:np.random.random((yDim,1)), directedEdges[jj-1]:np.random.random((yDim,1))}) # only one neigbour
                    self._edgeNeigboursConcensusVariable.update({directedEdges[jj+1]:np.random.random((yDim,1)),directedEdges[jj-1]:np.random.random((yDim,1))}) # only one neigbour 
                
                break # an agent will be present only once along one path    
                             
    def updateNeigboursConcensusVariables(self, edgeNeigbour: tuple[int,int], lagrancianCoefficient: np.ndarray, concensusVariable: np.ndarray) -> None :
        """update the concensus variables for a particular neigbour """
        
        if not (edgeNeigbour in self._edgeNeigboursLangrangianCoefficients) :
            raise ValueError(f"Given neigbour is not among the neigbours for this task. Given {edgeNeigbour}, available {list(self._edgeNeigboursConcensusVariable.keys())}")
        else :
            self._edgeNeigboursLangrangianCoefficients[edgeNeigbour] = lagrancianCoefficient
            self._edgeNeigboursConcensusVariable[edgeNeigbour] = concensusVariable
            
    def updateConcensusVariables(self,lagrancianCoefficient: np.ndarray, concensusVariable: np.ndarray)-> None :
        
        self._lagrangianCoefficient = lagrancianCoefficient
        self._concensusVariable     = concensusVariable
    
    
    def computeConcensusAverage(self):
        """computes current laplacian average of the y coefficients"""
        if not self._isIntialised :
            raise RuntimeError("Task container not yet iniitalised. Please iniitalise the container first")
        
        numNeigbours = len(self._edgeNeigboursConcensusVariable)
        concensusVariableNeigbourSum = 0
        
        for edge in self._edgeNeigboursConcensusVariable.keys() :
            concensusVariableNeigbour     = self._edgeNeigboursConcensusVariable[edge]
            concensusVariableNeigbourSum += concensusVariableNeigbour
    
        return numNeigbours*self._concensusVariable -  concensusVariableNeigbourSum # laplacian average of the ys
    

    def computeLagrangianCoefficientsAverage(self) :
        """computes current laplacian average of the lagarngina coefficients"""
        if not self._isIntialised :
            raise RuntimeError("Task container not yet iniitalised. Please iniitalise the container first")
        
        numNeigbours = len(self._edgeNeigboursLangrangianCoefficients)
        cNeigbourSum = 0
        
        for edge in self._edgeNeigboursLangrangianCoefficients.keys() :
            c = self._edgeNeigboursLangrangianCoefficients[edge]
            cNeigbourSum += c
    
        return numNeigbours*self._lagrangianCoefficient -  cNeigbourSum # laplacian average of the ys
        

    def correspondsTo(self,other)-> bool :
        """Check if a task or message have the same ID and edge of the orginal task container"""
        
        if not (isinstance(other,ConcensusVariableRequestMessage) or isinstance(other,ConcensusVariableResposeMessage) or isinstance(other,TaskContainer)) :
            raise ValueError("input must be a valid Concensus repose or concensus request")
        
        if isinstance(other,ConcensusVariableRequestMessage) or isinstance(other,ConcensusVariableResposeMessage) :
            sameTaskID = (other.taskID == self._taskID)
            sameEdge   = (other.edge == self.task.sourceTarget) or ((other.edge[1],other.edge[0]) == self.task.sourceTarget) # just matching the same edge
        else :
            sameTaskID = (other.taskID == self._taskID)
            sameEdge   = (other.task.sourceTarget == self.task.sourceTarget) or ((other.task.sourceTarget[1],other.task.sourceTarget[0]) == self.task.sourceTarget) # just matching the same edge
            
        
        if sameEdge and sameTaskID :
            return True
        else :
            return False
        
        
    
        
        
## Messages for concensus 

class ConcensusVariableRequestMessage():
    def __init__(self, senderID:int, edge:tuple[int,int], taskID:int) -> None:
        
        # agentID of the agent requiring the concensus variable
        self._taskID  = taskID   # taskID of the agent requiring the concensus variables
        self._senderID = senderID # agent ID if the agent requiring the concensus variable
        self._edge    = edge     # edge from which the concensus variable is requested
        
    @property
    def taskID(self):
        return self._taskID
    @property
    def edge(self):
        return self._edge
    @property
    def senderID(self):
        return self._senderID
    
      
    
    def __rrshift__(self, taskContainer: object) -> bool:
        """Check if a given TaskContainer corresponds to this request  message."""
        if not isinstance(taskContainer,TaskContainer) :
            raise ValueError(f"Cannot compare Task container with {taskContainer.__class__.__name__}")
        
        sameID   = taskContainer.taskID == self._taskID
        sameEdge = (taskContainer.task.sourceTarget == self._edge) or (taskContainer.task.sourceTarget == (self._edge[1],self._edge[0]))
        
        if sameID and sameEdge : # id the task container has same ID and same edge as the request
            return True
        else :
            return False
    


class ConcensusVariableResposeMessage() :
    def __init__(self, senderID:int,taskID:int,edge:tuple[int,int],lagrangianCoefficients:np.ndarray,concensusVariable:np.ndarray) -> None:
        # agentID of the agent of the agent sending the seponse
        self._taskID   = taskID
        self._senderID = senderID
        self._edge = edge
        self._lagrangianCoefficient  = lagrangianCoefficients #value of the lagrangian multipliers
        self._concensusVariable      = concensusVariable
        
    @property
    def taskID(self):
        return self._taskID
    @property
    def senderID(self):
        return self._senderID
    @property
    def edge(self):
        return self._edge
    
    @property
    def lagrangianCoefficient(self) :
        return self._lagrangianCoefficient
  
    @property
    def concensusVariable(self) :
        return self._concensusVariable

        
 
    
       
    def __rrshift__(self, taskContainer: object) -> bool:
        """Check if a given TaskContainer corresponds to this request  message."""
        if not isinstance(taskContainer,TaskContainer) :
            raise ValueError(f"Cannot compare Task container with {taskContainer.__class__.__name__}")
        
        sameID   = taskContainer.taskID == self._taskID
        sameEdge = (taskContainer.task.sourceTarget == self._edge) or (taskContainer.task.sourceTarget == (self._edge[1],self._edge[0]))
        
        if sameID and sameEdge : # id the task container has same ID and same edge as the request
            return True
        else :
            return False
        
    
class AgentTaskDecomposition() :
    """Agent Task Decomposition
        This calss provides a flexible framework for distributed task distribution in multi-agent systems
    """
    def __init__(self,agentID : int) -> None:
        """_summary_

        Args:
            agentID (int): Id of the agent
        """        
        
        self._optimizer              : ca.Opti              = ca.Opti() # optimizer for the single agent
        self._activeTaskContainers   : list[TaskContainer] = []        # list of active tasks that the agent has to include in its optiization prgram
        self._passiveTaskContainers  : list[TaskContainer] = []        # list of tasks to be aware of since the agent could be a supporting agent for these tasks
        
        self._agentID              : int                 =  agentID  # agent ID
        #self._penetrationVariables : list                = []        # auxiliarly variables for sets intersection
        
        self._warmStartSolution : ca.OptiSol   = None
        self._wasSolvedAlreadyOnce = False

        self._numIterations  = None
        self._currenetIt = 0
    
        self._penalty = self._optimizer.variable(1)
        self._optimizer.subject_to(self._penalty>=0)
        self._optimizer.set_initial(self._penalty,40)
        
        self._penaltyValues = []
        self._costValues = []
        self._isInitialisedForOptimization = False
        
    @property
    def agentID(self):
        return self._agentID

    @property
    def isInitialisedForOptimization(self):
        return self._isInitialisedForOptimization
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def taskContainers(self):
        return self._activeTaskContainers + self._passiveTaskContainers
    
    @property
    def activeTaskConstainers(self):
        return self._activeTaskContainers
    
    @property
    def passiveTaskConstainers(self):
        return self._passiveTaskContainers
    
    
    
    def _addSingleActiveTaskContainer(self, taskContainer : TaskContainer) :
        """_summary_

        Args:
            task (StlTask): STL task

        Raises:
            ValueError: if the the agentID is not the tareget or source of the given added task
            ValueError: if the given input is not an StlTask
        """        
        
        if not isinstance(taskContainer,TaskContainer) :
            raise ValueError(f"Input must be a valid instance of {TaskContainer.__class__.__name__}")
            
        else :
            taskContainer = copy.deepcopy(taskContainer) 
            task = taskContainer.task
            
            if not (self._agentID in task.sourceTarget) :
                raise ValueError(f"Trying to add a task with source target pair {task.sourceTarget} to agent with index {self._agentID}. This is not possible. Make sure that the task contains a sourceTraget pair matching the involved agents")
            else :
                self._activeTaskContainers.append(taskContainer)
            
    
    def _addSinglePassiveTaskContainer(self, taskContainer : TaskContainer) :
        """_summary_

        Args:
            task (StlTask): STL task

        Raises:
            ValueError: if the the agentID is not the tareget or source of the given added task
            ValueError: if the given input is not an StlTask
        """        
        
        taskContainer = copy.deepcopy(taskContainer)
        if not isinstance(taskContainer,TaskContainer) :
            raise ValueError(f"Input must be a valid instance of {TaskContainer.__class__.__name__}")
            
        else :
            task = taskContainer.task
            if not (self._agentID in task.sourceTarget) :
                raise ValueError(f"Trying to add a task with source target pair {task.sourceTarget} to agent with index {self._agentID}. This is not possible. Make sure that the task contains a sourceTraget pair matching the involved agents")
            else :
                self._passiveTaskContainers.append(taskContainer)
  
    
    def addActiveTaskContainers(self, taskContainers : TaskContainer|list[TaskContainer]) :
        """ list version of addSingleTask

        Args:
            tasks (TaskContainer|list[TaskContainer]): single or list of tasks
        """        
        
        if isinstance(taskContainers,list) :
            for taskContainer in taskContainers :
                self._addSingleActiveTaskContainer(taskContainer)
        else :
            self._addSingleActiveTaskContainer(taskContainers)
            
    def addPassiveTaskContainers(self, taskContainers : TaskContainer|list[TaskContainer]) :
        """ list version of addSingleTask

        Args:
            tasks (TaskContainer|list[TaskContainer]): single or list of tasks
        """        
        
        if isinstance(taskContainers,list) :
            for taskContainer in taskContainers :
                self._addSinglePassiveTaskContainer(taskContainer)
        else :
            self._addSinglePassiveTaskContainer(taskContainers)     
                
    def setUpOptimizer(self,numIterations :int) :
        """sets up the optimization problem for the given agent
        """    
        
    
        if len(self._activeTaskContainers)!=0 and (not self._isInitialisedForOptimization) :# initialised only once when you have active tasks
            self._numIterations     = int(numIterations) # number of iterations
            self._isInitialisedForOptimization  = True
    
            # here we need to set all the optimization variabled for your problem
            cost = 0
            for taskContainer in self._activeTaskContainers :
                
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
        sharedConstraint   = self._computeSharedConstraints()
        print("-----------------------------------")
        print(f"{self._agentID}")
        print(f"number of private constraints      : {len(privateConstraints)}")
        print(f"number of shared constraints       : {len(sharedConstraint)}")
        print(f"number of active  tasks            : {len(self._activeTaskContainers)}")
        print(f"number of passive tasks            : {len(self._passiveTaskContainers)}")
            
        if len(privateConstraints) != 0:
            self._optimizer.subject_to(privateConstraints) # it can be that no private constraints are present
        if len(sharedConstraint)!=0 :
            self._optimizer.subject_to(sharedConstraint)
        
        # set up cost and solver for the problem
        if  self._isInitialisedForOptimization :
            self._optimizer.minimize(cost)
            p_opts = dict(print_time=False, verbose=False)
            s_opts = dict(print_level=1)
    
            self._optimizer.solver("ipopt",p_opts,s_opts)
            self._cost = cost
        
    def _computeSharedConstraints(self) -> list[ca.Function]:
        """computes the shared inclusion constraint for the given agent. The shared constraints are the incluson of the path sequence of poytopes into the original decomposed polytope

        Returns:
            constraints (list[ca.Function]): set of constraints
        """        
    
        # select all the parametric tasks first
        parametericTasksContainers : list[TaskContainer] = [taskContainer for taskContainer in self._activeTaskContainers if taskContainer.task.isParametric]
        constraintSums  = []
        for taskContainer in parametericTasksContainers :
            
            task = taskContainer.task # extract task
            numComputingAgents = len(taskContainer.path)-1
            
            # For each task you have only 1 or two neigbours as you ae considering paths.
            # So you will only have one neigbour infront of the path and one neigbour behind.
            # Note that will remove this condition if we allow the agents to switch off the communication for the purpose of the decomposition, while we 
            # let the communication happen for the concensus. This is actually the best condition and we will need to consider that. 
            
            concensusVariableDim         = task.predicate.numHyperplanes*task.predicate.numVerices # dimension the concensus variable
            concensusVariableAveragePar  = self._optimizer.parameter(concensusVariableDim,1)                       # this is already the the average y parameter (N y_i - \sum_i^{N} y_j )
            concensusVariableValue       = np.zeros((concensusVariableDim,1))                                      # value of the concensus variable for the self agent initialised to zero for all the agents
            
           
            # for parameteric functions the center and scales are variable
            center : ca.MX = task.center
            scale  : ca.MX = task.scale
            originalCenter = task.originalCenter # center of the orginal predicate that has to be decomposed
            
            
            stackedConstraint = []
            for jj in range(task.predicate.numVerices) : # staking one constraint for each vertex
                vertex  = task.predicate.vertices[:,jj]
                stackedConstraint += [task.predicate.A@(center + scale*vertex) - (task.predicate.b + task.predicate.A@originalCenter)/numComputingAgents ]
               
            constraint1 = ca.vertcat(*stackedConstraint) - self._penalty   # yAveragePar = Ly laplacian concensus
            constraint = constraint1 + concensusVariableAveragePar <= 0
            # store the constraint, the parameter for the average concensus variable and the current value of the concensus variable into the task container
            taskContainer.saveConstraintParameters(constraint=constraint,concensusVariableAveragePar=concensusVariableAveragePar,concensusVariable=concensusVariableValue)
            constraintSums +=[constraint]
        
        return  constraintSums 
    
     
            
    def _computeOverloadingConstraints(self) -> list[ca.Function]:
        """mutiple collaborative tasks

        Returns:
            constraints (list[ca.Function]): resturns constraints for overloading of the edge with multiple collaborative tasks
        """        
        
        constraints = []
    
        if len(self._activeTaskContainers) == 1 :
           return  [] # with one single task you don't need overloading constraints
        
        taskCombinations : list[tuple[TaskContainer,TaskContainer]] = list(itertools.combinations(self._activeTaskContainers,2))
        
        for taskContainerI,taskContainerJ in taskCombinations :
            if not (taskContainerI.task.timeInterval  / taskContainerJ.task.timeInterval).isEmpty() : # there is an intersection
                constraints += self._computeIntersectionConstraint(taskContainerI,taskContainerJ)
   
        return constraints    
    
    def _computeIntersectionConstraint(self,taskContaineri:TaskContainer,taskContainerj:TaskContainer) -> list[ca.Function] :
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
        for taskContainer in self._activeTaskContainers :
            if (taskContainer.taskID == response.taskID) and (response.edge in taskContainer.edgeNeigboursConcensusVariable):
                
                taskContainer.updateNeigboursConcensusVariables(edgeNeigbour          = response.edge, 
                                                                lagrancianCoefficient = response.lagrangianCoefficient, 
                                                                concensusVariable     = response.concensusVariable) # update the neigbour concensus variable
                return 
        
        # check in the passive one
        for taskContainer in self._passiveTaskContainers :
            if (taskContainer.correspondsTo(response)):
                taskContainer.updateConcensusVariables(lagrancianCoefficient = response.lagrangianCoefficient, 
                                                       concensusVariable     = response.concensusVariable)
                return
                
        
    def replyConcensusRequest(self, request : ConcensusVariableRequestMessage) -> ConcensusVariableResposeMessage|None:
        """reply the concensus request"""
       
        # check first that you have the task among the active ones 
        for taskContainer in  self.taskContainers :
            
            # you only reply if you have a task container with the same edge as the requested one and the same task ID
            if taskContainer.correspondsTo(request):
                lagrangianCoefficient = taskContainer.lagrangianCoefficient
                concensusVariable     = taskContainer.concensusVariable
                response  = ConcensusVariableResposeMessage(senderID = self._agentID, 
                                                            taskID   = taskContainer.taskID,
                                                            edge     = request.edge,
                                                            lagrangianCoefficients = lagrangianCoefficient,
                                                            concensusVariable      = concensusVariable)
                return response # there is only one message per task, so you can break as soon as you found the one that you needed 
    
    def sendConcensusRequests(self) -> list[ConcensusVariableRequestMessage] :
        """Send a concensus request for the active tasks given to the agent"""
        requests : list[ConcensusVariableRequestMessage]  = []
        
        parametricActiveContainers  = [ taskContainer for taskContainer in self._activeTaskContainers if taskContainer.task.isParametric]
        parametricPassiveContainers = [ taskContainer for taskContainer in self._passiveTaskContainers if taskContainer.task.isParametric] 
        
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
        
        learningRate = 0.9 * (1/self._currenetIt)**0.7
        for taskContainer in self._activeTaskContainers : # update concensus variables active task containers
            if taskContainer.task.isParametric :
                cAverage = taskContainer.computeLagrangianCoefficientsAverage()
                taskContainer.concensusVariable = taskContainer.concensusVariable - learningRate*(cAverage) # concensus update
                
    def solveLocalProblem(self,printResult:bool=False) -> None :
        
        
        # algorithm start assuming the the others solved their problem (because you all set the lagangina multipliers to zero)
        
        for taskContainer in self._activeTaskContainers :
            if taskContainer.task.isParametric :
                yAverage = taskContainer.computeConcensusAverage()
                self._optimizer.set_value(taskContainer.concensusVariableAveragePar,yAverage)
                
        if self._warmStartSolution != None :
            self._optimizer.set_initial(self._warmStartSolution.value_variables())
        
        try :
            sol : ca.OptiSol = self._optimizer.solve() # the end 
        except  Exception as e: # work on python 3.x
            print("******************************************************************************")
            logging.error(f'The optimization for agent {self._agentID} failed with output: %s', e)
            print("******************************************************************************")
            print("The latest value for the variables was the following : ")
            print("penalty                 :",self._optimizer.debug.value(self._penalty))
            print("penetration coefficient :",self._optimizer.debug.value(self._t))
            print("cost                    : ",self._optimizer.debug.value(self._cost))
            print("******************************************************************************")
            
            exit()
      
        
        if printResult :
            print("--------------------------------------------------------")
            print(f"Agent {self._agentID} SOLUTION")
            print("--------------------------------------------------------")
            for taskContainer in self._activeTaskContainers :
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
        
        self._penaltyValues.append(self._optimizer.value(self._penalty))
        self._costValues.append(self._optimizer.value(self._cost))
        
        self._warmStartSolution = sol
        # update lagangian coefficients
        for taskContainer in self._activeTaskContainers :
            if taskContainer.task.isParametric :
                taskContainer.lagrangianCoefficient = self._optimizer.value(self._optimizer.dual(taskContainer.localConstraint))[:,np.newaxis]
        
        if not self._wasSolvedAlreadyOnce :
            self._wasSolvedAlreadyOnce = True
            self._optimizer.solver("sqpmethod",{"qpsol":"qpoases"}) 
            
        
        # save current iteration
        self._currenetIt += 1
        
    def getListOfActiveTasks(self) -> list[StlTask]:
        """
           Returns the list of active tasks for this agent. Indeed, the active tasks are all defined over one edge and they are the tasks that the agent needs to use to solve the parameters of the parameteric tasks. 
           Note that both parameteric and non parametric tasks are present here since both need to be used in the optimization. The returned list of tasks does not contain any parameteric task since the parameter were found as a resul of the 
           optimization.
        """

        # tasks that are not parameteric over the edge
        nonParametricTasks = [taskContainer.task for taskContainer in self._activeTaskContainers if not taskContainer.task.isParametric]
        
        solvedTasks = []
        if self._wasSolvedAlreadyOnce : # if there was an optimization
            for taskContainer in self._activeTaskContainers : # look in all the active tasks of the edge
                if taskContainer.task.isParametric : # check the ones that are parameherrioc and hence needs to be replaced by nonParametrci tasks after the optimzation
                    center = self._optimizer.value(taskContainer.task.center)
                    scale = self._optimizer.value(taskContainer.task.scale)
                    taskContainer.task.predicate.A
                    # create a new task out of the parameteric one
                    predicate = PolytopicPredicate(A     = taskContainer.task.predicate.A,
                                                         b     = scale*taskContainer.task.predicate.b,
                                                         center= center)
                    task = StlTask(temporalOperator   = taskContainer.task.temporalOperator,
                                         timeinterval       = taskContainer.task.timeInterval,
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
      self._taskContainers = []
      
      if (not isinstance(source,int)) or (not isinstance(target,int)) :
          raise ValueError("Target source pairs must be integers")
      else :
          self._sourceNode = source
          self._targetNode = target
      
    @property
    def tasksList(self) :
        return self._tasksList
    @property
    def taskContainers(self):
        return self._taskContainers
    
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
    
   
    def _addSingleTask(self,inputTask : StlTask|TaskContainer) -> None :
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not (isinstance(inputTask,StlTask) or isinstance(inputTask,TaskContainer)) :
            raise Exception("please enter a valid STL task object or a list of StlTask objects")
        else :
            if isinstance(inputTask,StlTask) :
                # set the source node pairs of this node
                if inputTask.hasUndefinedDirection :
                    inputTask.sourceTarget(source=self._sourceNode,target=self._targetNode)
                    self._taskContainers.append(TaskContainer(task=inputTask))
                    self._tasksList.append(inputTask) # adding a single task
                else :
                    if ((self._sourceNode,self._targetNode) != (inputTask.sourceNode,inputTask.targetNode)) and  ((self._targetNode,self._sourceNode) != (inputTask.sourceNode,inputTask.targetNode)):
                        raise Exception(f"Trying to add a task with defined source/target pair to an edge with different source/target pair. task has source/tarfet pair {inputTask.sourceTarget} and edge has {(self._sourceNode,self.targetNode)}")
                    else :
                        self._taskContainers.append(TaskContainer(task=inputTask))
                        self._tasksList.append(inputTask) # adding a single task
            
            
            elif isinstance(inputTask,TaskContainer):
                # set the source node pairs of this node
                task = inputTask.task
                if task.hasUndefinedDirection :
                    task.sourceTarget(source=self._sourceNode,target=self._targetNode)
                    self._taskContainers.append(inputTask) # you add directly the task container
                    self._tasksList.append(task) # adding a single task
                else :
                    if ((self._sourceNode,self._targetNode) != (task.sourceNode,task.targetNode)) and  ((self._targetNode,self._sourceNode) != (task.sourceNode,task.targetNode)):
                        raise Exception(f"Trying to add a task with defined source/target pair to an edge with different source/target pair. task has source/tarfet pair {task.sourceTarget} and edge has {(self._sourceNode,self.targetNode)}")
                    else :
                        self._taskContainers.append(inputTask)
                        self._tasksList.append(task) # adding a single task
            
    
    def addTasks(self,tasks : StlTask|TaskContainer|list[StlTask]|list[TaskContainer]):
        if isinstance(tasks,list) : # list of tasks
            for  task in tasks :
                self._addSingleTask(task)
        else :# single task case
            self._addSingleTask(tasks)
    
 
    def flagOptimizationInvolvement(self) -> None :
        self._isInvolvedInOptimization = 1
        
    def cleanTasks(self)-> None :
        del self._taskContainers
        del self._tasksList
        
        self._taskContainers = []
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
            tasksContainersToBeDecomposed: list[TaskContainer] = edgeObj.taskContainers
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
                    subtask = createParametericTaskFrom(task = task,source = sourceNode,target = targetNode) # creates a parameteric copy of the original task
                    subTaskContainer = TaskContainer(task = subtask,taskID = taskContanier.taskID ,path = path)
                    subtaskEdge.addTasks(subTaskContainer) 
    
        
    optimizationEdges = [] 
    for path in  pathList :
        optimizationEdges += edgeSet(path)
    
    optimizationEdges = list(set(optimizationEdges)) # unique set of edge unsed for the optimization
    numOptimizationIterations = 1000
    
    
    for edgeObj in edgeList :
   
        if edgeObj.isCommunicating and edgeObj.hasSpecifications : # for all the cummincating agent start putting the tasks over the nodes
  
            decompositionAgents[edgeObj.sourceNode].addActiveTaskContainers(edgeObj.taskContainers)  # all the containers for the optimization (source node is the one makijng computations actively on this constraints)
            decompositionAgents[edgeObj.targetNode].addPassiveTaskContainers(edgeObj.taskContainers) # all the containers for message resumbission (target node willl forward this information if necessary to other nodes)
               
    # after all constraints are set, initialise source nodes as computing agents for the dges that were used for the optimization
    for agent in decompositionAgents.values() :
        agent.setUpOptimizer(numIterations=numOptimizationIterations)
    
    concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents)# share current value of private lagrangian coefficients and auxiliary y variable
    # find solution
    for jj in range(numOptimizationIterations) :
        for agentID,agent in decompositionAgents.items() :
            if agent.isInitialisedForOptimization : #only computing agents should make computations 
                agent.solveLocalProblem()
        concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents)
        # update y
        for agentID,agent in decompositionAgents.items() :
            if agent.isInitialisedForOptimization :
                agent.updateY() # update the value of y
        concensusRound(commGraph=commGraph,decompositionAgents=decompositionAgents) # concensus before leaving

    fig,(ax1,ax2) = plt.subplots(1,2)
    for agentID,agent in decompositionAgents.items() :
        if agent.isInitialisedForOptimization :
            ax1.plot(range(numOptimizationIterations),agent._penaltyValues,label=f"Agent :{agentID}")
            ax1.set_ylabel("penalties")
            
            ax2.plot(range(numOptimizationIterations),agent._costValues,label=f"Agent :{agentID}")
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
    decompositionDictionay : dict[TaskContainer,dict[tuple[int,int],TaskContainer]]= {}
    
    # select the agents that have actually made computations 
    computingAgents = {agentID:agent for agentID,agent in decompositionAgents.items() if agent.isInitialisedForOptimization}
    
    
    for path in decompositionPaths :
        # now take initial and final index since this is the non communicating edge 
        (i,j) = path[0],path[-1]
        edgeObj = findEdge(edgeList = edgeObjectsLists, edge =(i,j)) # get the corresponding edge
        decomposedTasks : list[TaskContainer] = edgeObj.taskContainers                                # get all the tasks defined on this edge
        
        for taskContainer in decomposedTasks :
            tasksAlongThePath = {}
            for agentID,agent in computingAgents.items() :
                agentTaskContainers = agent.activeTaskConstainers # search among the containers you optimised for 
                for container in agentTaskContainers : # check which task on the agent you have correspondance to
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
        print(f"Temporal operator : {decomposedTaskContainer.task.temporalOperator}")
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
        ax[counter].set_title(f"Task edge {decomposedTaskContainer.task.sourceTarget}\n Operator {decomposedTaskContainer.task.temporalOperator} {decomposedTaskContainer.task.timeInterval}")
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
        activeTaskContainers = agent.activeTaskConstainers # active task containers
        passiveTaskContainers = agent.passiveTaskConstainers # active task containers
        
        for taskContainer in activeTaskContainers :
            if taskContainer.task.isParametric :
                center = agent.optimizer.value(taskContainer.task.center)
                scale  = agent.optimizer.value(taskContainer.task.scale)
                
                taskContainer.task.predicate.A
                # create a new task out of the parameteric one
                predicate = PolytopicPredicate(A     = taskContainer.task.predicate.A,
                                                     b     = scale*taskContainer.task.predicate.b,
                                                     center= center)
                task = StlTask(temporalOperator   = taskContainer.task.temporalOperator,
                                     timeinterval       = taskContainer.task.timeInterval,
                                     predicate          = predicate,
                                     source             = taskContainer.task.sourceNode,
                                     target             = taskContainer.task.targetNode,
                                     timeOfSatisfaction = taskContainer.task.timeOfSatisfaction)
                taskList += [task]
                
                
            else : # no need for any parameter
                taskList += [taskContainer.task]
                        
        for taskContainer in passiveTaskContainers :
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
                task = StlTask(temporalOperator   = taskContainer.task.temporalOperator,
                                     timeinterval       = taskContainer.task.timeInterval,
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
