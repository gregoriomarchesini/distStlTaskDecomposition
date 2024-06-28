from .decomposition_module import *
from .predicate_builder_module import *
import casadi as ca

class STLController():
    """STL QP based controller class for a 2D Holonomic robot"""
    def __init__(self,agentIndex : int,neighbours:list[int],alphaMarginOnBarrier = 0) -> None:
        
        self._StlTasks           : list[StlTask]    =  []             # list of all the tasks to be satisfied
        self._agentIndex         : int              = agentIndex
        self._opti               : ca.Opti          = ca.Opti('conic')
        self._neighbours         : list[int]        = neighbours       # list of neigbouring agents in the communication graph
        self._stateSpaceDim      : int              = 2               # state dimensioin assumed to be 2D
        
        # all agents MUCT have the same collsion parameters. Otherwise something unexpected could happen
        self._maxNumCollisionChecks : int      = 10
        self._sensingRadius         :float     = 2.              # we set up a set of 10 parametric constraints. These can be used to avoid collisons with up to 10 agents in a radius of 10 m
        self._collisionDistance     :float     = 1.        
        
        
        self._barrierConstraints   =  []         # barrier constraint
        self._barrierDerivative    =  []
        self._barrierValues       =  []
        self._activationFunctions =  []
        self._collsionConstraints =  []
        
        self._nominalControlInput = 0
        
        self._initializationTime = 0
        
        if alphaMarginOnBarrier>= 0 : # alpha margin imposed when setting the barrier constraint as dot(b) + alpha(b) >= alphaMarginOnBarrier
            self._alphaMarginOnBarrier = alphaMarginOnBarrier #  non-negative 
        else :
            raise ValueError("Alpha margin should be non negative")
        
        
        self._previouslyInitalised = False
        
        
        # variables applied for functions creation
        self._timeVar                    = ca.MX.sym("dd",1)                    # time variable to be applied
        self._agentStateVar              = ca.MX.sym("xi",self._stateSpaceDim,1) # state variable for the agents
        self._neigbourStateVar           = ca.MX.sym("xi",self._stateSpaceDim,1) # state variable for the agents
        self._dummyScalar                = ca.MX.sym("scale",1)                 # it will serve to put the minimum approximation at a postive value
        self._controlVar                 = ca.MX.sym("xi",self._stateSpaceDim,1) # state variable for the agents
        
        self._dynamics = self._controlVar
        
        self._control  = self._opti.variable(self._stateSpaceDim,1) # taskes the dimension of the control input
        
        # set up all the parameters for optimization
        self._currentAgentStatePar       = self._opti.parameter(self._stateSpaceDim ,1)
        self._commNeighboursStatePar      = {neigbour:self._opti.parameter(self._stateSpaceDim,1) for neigbour in self._neighbours} # for each of your neighbours you create a state parameter. This parameter willm then be used to be replaced in the optimization program
        self._tPar                       = self._opti.parameter(1)
        self._counter                    = 0
        
        self._collsionsAvidanceParameters = {jj:(self._opti.parameter(1),self._opti.parameter(self._stateSpaceDim,1)) for jj in range(self._maxNumCollisionChecks) } # they key is gonna be a bit (0/1) and the item is a state parameter. Every time a possible collsion is detercted we set the bit to 1 and the state is used in a cllision avoidance constraint.
        
        # current control input for reference
        self._currentControlInput = np.zeros(self._stateSpaceDim) # initial control input assumed
        self._warmstartSol = None
        
        
    def _addSingleTask(self,task : StlTask) -> None:  
        """add tasks to the task list to be satsified by the controller"""
        if isinstance(task,StlTask) :
                # set the source node pairs of this node
                if (task.sourceNode != self._agentIndex) and (task.targetNode != self._agentIndex) :
                    raise ValueError(f"Seems that one or more of the inserted tasks do not involve the state of the current agent. Indeed agent index is {self._agentIndex}, but task is defined over edge ({task.sourceTarget})")
               
                elif task.isParametric:
                    raise RuntimeError("Seems you are trying to input a parameteric task in the controller. This is not supported. Please solve the decomposition problem for the parameteric task and input the resulting task as a non parameteric task")
                   
                else :
                    self._StlTasks.append(task) # adding a single task  
        else :
            raise Exception("please enter a valid STL task object or a a list of StlTask objects")
        
        
        
    def addTasks(self,tasks : list[StlTask]|StlTask) -> None :
        
        if isinstance(tasks,list) :
            for task in tasks :
                self._addSingleTask(task=task)
        else :
            self._addSingleTask(task)
    
    
    def _setConstraints(self,initialNeighboursState:dict[(int,np.ndarray)],initialAgentState: np.ndarray):
        """set up controller constraints

        Args:
            initialNeighboursState (dict[(int,np.ndarray)]): dictionary of initial states for the agents you are connected to
            initialAgentState     (np.ndarray)            : your initial state

        Raises:
            ValueError         : if keys  in initialNeighboursState are not matching the neighbours for the agent
            NotImplementedError:
        Note :
        
        The initial agent state is required because the agent has to intialise the inital value of the time decaying function gamma(t). This funciton will determine how the time evolution of the barrier function goes
         
        """     
        
        for key in initialNeighboursState :
            if not key in self._neighbours :
                raise ValueError("at least one of the neighbours given is not in the neighbours list. Please update the neighbours list")
        
        if len(self._StlTasks) == 0: # no tasks to be fullfilled
            self._barrierConstraints = []
            return 
        
        for task in self._StlTasks :
            self._addBarrierConstraint(initialAgentState = initialAgentState,initialNeighboursState = initialNeighboursState,task = task)
        
        # self._setCollsionAvoidanceConstraint() # collision constraints
        
        # maximum control input 
        
        self._previouslyInitalised = True       
    
    
    def _addBarrierConstraint(self,initialAgentState:np.ndarray,initialNeighboursState:dict[int,np.ndarray],task:StlTask) :
        """_summary_

        Args:
            initialAgentState (np.ndarray)              : initial state of the agent
            initialNeighboursState (dict[int,np.ndarray]): initial state of each neigbouring agent
            task (StlTask)                              : task associated with the predicate 

        Raises:
            ValueError: _description_
            RuntimeError: _description_
        """        
         
        self._counter +=1
        
         # extract information about the barrier direction     
        target = task.targetNode
        source = task.sourceNode 
        timeSatisfaction = task.timeOfSatisfaction # time at which the task is initially sastified
        timeOfRemotion   = task.timeOfRemotion     # time at which the barrier can be removed since it was already satisfied
        
        # extract the predicate information from the task RECALL -> (A(x-center)-z<=0)
        A      = task.predicate.A
        b      = task.predicate.b
        center = task.predicate.center
        chebRadius = task.predicate.polytope.chebR
        chebCenter = task.predicate.polytope.chebXc
        if np.ndim(chebCenter) == 1:
            chebCenter = chebCenter[:,np.newaxis]
        
        
        # add nominal controller
        if target == source : # single agent task
                self._nominalControlInput += -(self._currentAgentStatePar - center)/(timeSatisfaction-self._initializationTime)*  ca.if_else(self._tPar<=timeSatisfaction,1,0)
        else :
            if self._agentIndex == target : # if you are the target
                self._nominalControlInput += -((self._currentAgentStatePar-self._commNeighboursStatePar[source])  - center)/(timeSatisfaction-self._initializationTime)   *  ca.if_else(self._tPar<=timeSatisfaction,1,0)
            else : # reveresed edge (if you are the source)
                self._nominalControlInput += -((self._commNeighboursStatePar[target]-self._currentAgentStatePar)  - center)/(timeSatisfaction-self._initializationTime)*  ca.if_else(self._tPar<=timeSatisfaction,1,0)
        
        (rows,cols) = np.shape(A)
        # set all dimensions correctly
        if np.ndim(b) == 1:
            b = b[:,np.newaxis]
        if np.ndim(center) ==1 :
            center = center[:,np.newaxis]
        
        # for barrier functions we want (A(x-center)-z>=0). Hence we change the sign
  
        linearInequalities =  -1* ( A@ (self._agentStateVar-center) - b)
     
        predicates = []
        for jj in range(rows): # make a predicate out of each hypeplrane defyning the convex set
            predicates += [ca.Function("predicateFunction",[self._agentStateVar],[linearInequalities[jj]])]
        
        # if you want to approximate the convex sets by a single inner approximation through chebychev circles
        # predicates = [ca.Function("predicateFunction",[self._agentStateVar],[chebRadius**2 - (self._agentStateVar-chebCenter).T@(self._agentStateVar-chebCenter)])]
        
        # create alpha funciton
        alpha    = ca.Function("quadratic",[self._dummyScalar],[ca.if_else((self._dummyScalar)<=0,50*self._dummyScalar,0.2*self._dummyScalar)]) # non-smooth alpha. When negative it provides incentive to go further toward the satisfaction of the barrier

        
        # REMEMBER : a task is given by (A(c-x_i)-b<=0) or (A(c-e_{ij})-b<=0). So in one case you have the edge vector and in the other case you have the state vector of a single agent. 
        # we now compute the initial value of the barrier for all the cases
        
        predicatesInitialValues = []
        for predicate in predicates :
            if target == source : # single agent task
                predicatesInitialValues += [predicate(initialAgentState)] # initial value of the barrier. At the beginning it is commonly negative because you do not satisfy the barrier at the beginning. So this is the reason of the minus sign
            else :
                if self._agentIndex == target : # if you are the target
                
                    initialEdgeState = initialAgentState - initialNeighboursState[source]
                    predicatesInitialValues += [predicate(initialEdgeState)]
                    
                else : # reveresed edge (if you are the source)
                    
                    initialEdgeState = initialNeighboursState[target] - initialAgentState
                    predicatesInitialValues += [predicate(initialEdgeState)]
            
            
        # for each barrier we now create a time transient function :
        barriers = []
        scatterFactor = 1.1
        timeScatterFactor = 1
        
        for predicate,predicateInitialValue in zip(predicates,predicatesInitialValues) :
            
            scaleFactor =  scatterFactor*(1+(timeSatisfaction-self._initializationTime)/50)   # >=1 # margin over the satisfaction of the barrier. Barriers with more urgent time of satisfaction will get smaller scale faactors. This is to avoid sharp decays
            scatterFactor += 0.6
            
            if predicateInitialValue <= 0 : 
                if timeSatisfaction==0 :
                    raise ValueError("There is at least one task which should be satsified from time t=0, but the intial state of the agent(s) is already outside the satisfying region. please veryfy that this is the case")
                gamma0  = scaleFactor*-predicateInitialValue # we use (1-predicateInitialValue) so that we have at least a value of 1. Indeed if predicateInitialValue is smaller than 1, then the scale factor will have little effect we gust need a gamma0 such that gamm0 + predicateInitialValue>=0. 
            else :
                gamma0  = scaleFactor*predicateInitialValue #scaleFactor*predicateInitialValue # in case the predicate is positive it is good to give some margin anyway. This will give some freedoms to satisfy other tasks in the meanwhiel
            
            if timeSatisfaction<=self._initializationTime :
                print(f"fomula with id:{id(task)} cannot be satisfied as the time of satisfaction is passed the initalization time. Initalization time is {self._initializationTime} and given satisfaction time is {timeSatisfaction} ")
                print("the tasks has the following specifics :")
                print(f"time interval : {task.time_interval.a,task.time_interval.b}")
                print(f"temporal operator : {task.temporal_operator}")
                raise RuntimeError("Unsatisfiable task")
        
            # timeSatisfaction = np.max([0,timeSatisfaction - timeScatterFactor]) # scattering the time of the barriers for staisfaction. Helps break symmetris
            slope    = -gamma0/(timeSatisfaction-self._initializationTime) # negative slope
            gamma    = ca.if_else(self._timeVar<=timeSatisfaction,gamma0+(self._timeVar-self._initializationTime)*slope,0) # piece wise linear function
            # gammaDot = ca.if_else(self._timeVar<=timeSatisfaction,slope,0) # derivative of gamma already given explicitly for the barrier constraint
            activationFunction = ca.if_else(self._timeVar<=timeOfRemotion*1.1,1.,0.) # sets the constrant directly to zero when it is not required anymore
        
            barrier  = ca.Function("barrierFunction",[self._agentStateVar,self._timeVar],[predicate(self._agentStateVar) + gamma])
            barriers += [barrier] # adding the barriers 
        
        # at this point we have create  a barrier for each hyperplane in the convex polytope of the task. Now time to construct the barrier constrant 
        #  db(x,t)/dt + nabla b(x,t) f(x) + g(x)u + alpha(b(x,t))>=0.  (for the singel integrator we have just  db(x,t)/dt + nabla b u + alpha(b(x,t))>=0.)
        
        for barrier in barriers :
            if source == target :  # single agent specification
                
                nablaXi = ca.jacobian(barrier(self._agentStateVar,self._timeVar),self._agentStateVar) # recall b(x,t) = gamma(t) + h(x) -> nabla b(x,t) = nabla h(x)
                dbdt    = ca.jacobian(barrier(self._agentStateVar,self._timeVar),self._timeVar) 
                
                # bdot(x,t) + nabla_xi b(x,t)^T * u + alpha(b(x,t))) >=0
                barrierCondition  = ca.Function("barrierCondition",[self._agentStateVar,self._timeVar,self._controlVar],
                    [ activationFunction*( ca.dot(nablaXi.T, self._dynamics) + (dbdt + alpha(barrier(self._agentStateVar,self._timeVar)))) ])
                
                localDerivative = ca.Function("barrierCondition",[self._agentStateVar,self._timeVar,self._controlVar],
                    [ activationFunction*( ca.dot(nablaXi.T, self._dynamics)) ])
                
                self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._tPar,self._control) - self._alphaMarginOnBarrier] # parameteric constraint
                self._barrierValues      += [barrier(self._currentAgentStatePar,self._tPar)]
                self._barrierDerivative  += [localDerivative]
            
            else : # case of the edge 

                if target == self._agentIndex :
                    eijVar  = self._agentStateVar - self._neigbourStateVar
                    
                    nablaXi = ca.jacobian(barrier(eijVar,self._timeVar),self._agentStateVar)
                    dbdt    = ca.jacobian(barrier(eijVar,self._timeVar),self._timeVar) 
                    
                    # compute nablas for the barriers
                    nablaXNorm  = ca.norm_2(ca.jacobian(barrier(eijVar,self._timeVar),ca.vertcat(self._agentStateVar,self._neigbourStateVar)))**2
                    nablaXiNorm = ca.norm_2(ca.jacobian(barrier(eijVar,self._timeVar),self._agentStateVar))**2
                    
                    # load sharing function
                    loadSharingFunction = ca.if_else(nablaXNorm>=0.0000001,nablaXiNorm/nablaXNorm,0)
                    
                    # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                    barrierCondition    = ca.Function("barrierCondition",[self._agentStateVar,self._neigbourStateVar,self._timeVar,self._controlVar],[
                        activationFunction*( ca.dot(nablaXi.T, self._dynamics) + ( dbdt + alpha(barrier(eijVar,self._timeVar)))*loadSharingFunction)])
                    
                    # now take the gradient of the function 
                    self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._commNeighboursStatePar[source],self._tPar,self._control)] # para
                    self._barrierValues      += [barrier(self._currentAgentStatePar-self._commNeighboursStatePar[source],self._tPar)]
                
                else :
                    
                    eijVar  = self._neigbourStateVar - self._agentStateVar # note how here it is reversed
                    
                    nablaXi = ca.jacobian(barrier(eijVar,self._timeVar),self._agentStateVar)
                    dbdt    = ca.jacobian(barrier(eijVar,self._timeVar),self._timeVar) 
                    
                    # compute nablas for the barriers
                    nablaXNorm  = ca.norm_2(ca.jacobian(barrier(eijVar,self._timeVar),ca.vertcat(self._agentStateVar,self._neigbourStateVar)))**2
                    nablaXiNorm = ca.norm_2(ca.jacobian(barrier(eijVar,self._timeVar),self._agentStateVar))**2
                    
                    # load sharing function
                    loadSharingFunction = ca.if_else(nablaXNorm>=0.0000001,nablaXiNorm/nablaXNorm,0)
                    
                    # bdot(x,t) + (nabla_xi b(x,t)^T * u + alpha(b(x,t)))eta_sharing >=-0
                    barrierCondition    = ca.Function("barrierCondition",[self._agentStateVar,self._neigbourStateVar,self._timeVar,self._controlVar],[
                        activationFunction*( ca.dot(nablaXi.T, self._dynamics) + ( dbdt + alpha(barrier(eijVar,self._timeVar)))*loadSharingFunction)])
            
                    self._barrierConstraints += [barrierCondition(self._currentAgentStatePar,self._commNeighboursStatePar[target],self._tPar,self._control)]
                    self._barrierValues      += [barrier(self._commNeighboursStatePar[target]-self._currentAgentStatePar,self._tPar)]


    def _setCollsionAvoidanceConstraint(self) :
        # non-collaborative collsion avoidance ! (you assume that the other agent will not help with the barrier function)
        # barrier    : h(x1,x2) = ((x1-x2)**2 - d**2) >=0
        # constraint : (x1-x2)(u1-u2) + ((x1-x2)**2-d**2) >=0 # both the agents try to satsify this condition independently, (assumeed alpha function to be 1)
        # agent 1 has (x1-x2)u1 + ((x1-x2)**2 - d**2)>=0
        # agent 2 has (x2-x1)u2 + ((x1-x2)**2 - d**2)>=0f
        
        for jj,(collsionBitParameter,statePar) in self._collsionsAvidanceParameters.items():
            # we add a simple barrier collision avoidance constrains for the two agents
            self._collsionConstraints  += [ collsionBitParameter *  ((self._currentAgentStatePar - statePar).T@self._dynamics + (ca.norm_2(self._currentAgentStatePar - statePar)**2- self._collisionDistance**2)) >=0]
        
        
        # now adding collision avoidance constrainsts
        self._opti.subject_to(self._collsionConstraints)
 
    def setUp(self,initialNeighboursState:dict[(int,np.ndarray)],initialAgentState:np.ndarray,initializationTime:float,allowSlackSatisfaction : bool= False) -> None:
        
        
        self._epsilonsSlack  = [] 
        cost = 0
        self._initializationTime = initializationTime
        if self._previouslyInitalised :
            self._clean() # after the first time 
        
        self._setConstraints(initialNeighboursState = initialNeighboursState , initialAgentState = initialAgentState)
        # cost += (0.01*(self._nominalControlInput.T/(ca.norm_2(self._nominalControlInput)+0.001)  @ self._control))**2
        cost += 3*self._control.T@self._control 
        # adding barrier constraints
        if allowSlackSatisfaction :
            if not len(self._StlTasks)==0 :
                # adding barrier constrant here
                for jj,constraint in enumerate(self._barrierConstraints) :
                    
                    epsilon = self._opti.variable(1)
                    self._opti.set_initial(epsilon,0)
                    self._opti.subject_to(epsilon>=0) 
                    
                    self._opti.subject_to(constraint >=-epsilon) # add all the constraints
                    self._epsilonsSlack.append(epsilon)
                    cost += 100*epsilon # high cost fo violating the task
                    
        else :
            if not len(self._StlTasks)==0 :
                for constraint in self._barrierConstraints :
                    self._opti.subject_to(constraint >=0) # add all the constraints
                cost += 0
        
        
        self._opti.minimize(cost) 
        self._opti.solver("qpoases",{"printLevel":"none"})
        
        # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        # self._opti.solver("ipopt",opts)
        
    def computeControlInput(self,currentCommNeighboursState:dict[int,np.ndarray],currentAgentState:np.ndarray,time:float,collidingNeighbours:list) :
        """_summary_

        Args:
            currentNeighboursState (dict[int,np.ndarray]): state of the neighbours for that communicate their state to you in order to get a task satsified together
            currentAgentState (np.ndarray): current state of the agent
            time (float): current time
            collidingNeighbours (list): this is the state of the agents that are not communicating with you but that are possible candidates for a collision

        Raises:
            ValueError: if there is a neigbour trying to communicate with the agent that is not among the neighbours list

        Returns:
            _type_: _description_
        """      
        
        
        
        if not len(self._StlTasks)== 0 :
            for key,neigbourParametericState in self._commNeighboursStatePar.items() :
                try :
                    self._opti.set_value(neigbourParametericState,currentCommNeighboursState[key]) # assign each paramatric state to the neighbouring agents
                except KeyError :
                    raise ValueError("One or many of the neighbours give are not in the neighbours list. Please verify that the state of the neighbours given are correct")
            
        
            self._opti.set_value(self._currentAgentStatePar,currentAgentState)
            self._opti.set_value(self._tPar,time)
            
        if len(collidingNeighbours)==0 :
            for collisionBit,statePar in self._collsionsAvidanceParameters.values():
                self._opti.set_value(collisionBit,0)
                self._opti.set_value(statePar,np.array([0,0]))
                
        else : # case in which you have sensed some possible neigbour
            
            # set the parameters for all the neighbours you receive
            for jj,neigbourState in enumerate(collidingNeighbours) :
                
                collisionBit,statePar = self._collsionsAvidanceParameters[jj]
                self._opti.set_value(collisionBit,1)
                self._opti.set_value(statePar,neigbourState)
            
            # set the other parameters to zero so that no constraint is needed
            for kk in range(jj+1,self._maxNumCollisionChecks) :
                collisionBit,statePar = self._collsionsAvidanceParameters[kk]
                self._opti.set_value(collisionBit,0)
                self._opti.set_value(statePar,np.array([0.,0.]))
             
             
        if self._warmstartSol != None : 
            self._opti.set_initial(self._warmstartSol)   
            
    
        sol = self._opti.solve()
        self._currentControlInput = sol.value(self._control)
        self._warmstartSol = sol.value_variables()
       
        
        return self._currentControlInput
    
    
    def _cleanOutdatedTasks(self) :
        self._StlTasks = [task for task in self._StlTasks if not(task.timeOfRemotion<self._initializationTime)]
    
    def _defineNewOptimizer(self) :
        
        # clean the optimizer
        del self._opti 
        self._opti =  ca.Opti('conic')
        
        # set up all the parameters for optimization
       
        self._tPar                   = self._opti.parameter(1)
        self._counter = 0
        self._epsilonsSlack          = []
        
        self._currentControlInput = np.zeros(self._stateSpaceDim)
        
        self._barrierConstraints  =  []         # barrier constraint
        self._barrierGradients    =  []
        self._barrierValues       =  []
        self._activationFunctions =  []
        self._collsionConstraints =  []
        
        self._control  = self._opti.variable(self._stateSpaceDim,1) # taskes the dimension of the control input
        
        # set up all the parameters for optimization
        self._currentAgentStatePar       = self._opti.parameter(self._stateSpaceDim ,1)
        self._commNeighboursStatePar      = {neigbour:self._opti.parameter(self._stateSpaceDim,1) for neigbour in self._neighbours} # for each of your neighbours you create a state parameter. This parameter willm then be used to be replaced in the optimization program
        self._tPar                       = self._opti.parameter(1)
        self._counter                    = 0
        
        self._collsionsAvidanceParameters = {jj:(self._opti.parameter(1),self._opti.parameter(self._stateSpaceDim,1)) for jj in range(self._maxNumCollisionChecks) } # they key is gonna be a bit (0/1) and the item is a state parameter. Every time a possible collsion is detercted we set the bit to 1 and the state is used in a cllision avoidance constraint.
        
        # current control input for reference
        self._currentControlInput = np.zeros(self._stateSpaceDim) # initial control input assumed
        self._warmstartSol = None
        self._nominalControlInput = 0
        
    def _clean(self):
        self._cleanOutdatedTasks()
        self._defineNewOptimizer()
        




class Agent() :
    def __init__(self,initialState : np.ndarray,agentID : int,neighbours) -> None:
        
        self._initialState  :np.ndarray = initialState
        self._agentID       : int = agentID
        self._neighbours    :list = neighbours
        
        self._currentState          = initialState
        self._currentCommNeighboursState = dict()
        self._deltaT                = 0.01
        self._alphaMargin  =  10*self._deltaT # 
        
        self._controller   = STLController(agentIndex=self._agentID,neighbours=self._neighbours,alphaMarginOnBarrier=self._alphaMargin)
        
        stateVar   = ca.MX.sym("stateVar",len(initialState))
        controlVar = ca.MX.sym("stateVar",len(initialState))
        
        self._dynamics  = ca.Function("singleIntergator",[stateVar,controlVar],[stateVar + self._deltaT*controlVar]) #TODO: we will have to do something more fancy. Note that also the controller will have to get this dynamics diuring the set up
        
        self._neighboursStatesUpdated = True
        
    @property
    def currentState(self):
        return self.currentState
    @property
    def currentControlInput(self) :
        return self._currentControlInput
    @property
    def timeStep(self):
        return self._deltaT
    
    @property
    def neighbours(self):
        return self._neighbours 
    @property
    def agentID(self):
        return self._agentID
    
    
    def initializeController(self,initialNeighboursState : dict,tasks : list[StlTask],initializationTime:float,allowSlackSatisfaction = False) :
        
        
        self._currentCommNeighboursState            = initialNeighboursState
        self._controller.addTasks(tasks = tasks)
        self._controller.setUp(initialNeighboursState = initialNeighboursState,
                               initialAgentState      =  self._initialState,
                               initializationTime     = initializationTime,
                               allowSlackSatisfaction = allowSlackSatisfaction) 
        
        self._hasTasks = len(self._controller._StlTasks)
        
    
    def cleanController(self,initialNeighboursState : dict,initializationTime:float,allowSlackSatisfaction = False) :
        """This need to be called when you have already ba controller setted up"""
        if not self._controller._previouslyInitalised :
            raise Exception("controller for the urrent agent was not initalized before. Please run a first initialization before cleaning the controller")
        
        
        self._intializationTime = initializationTime
        self._currentCommNeighboursState = initialNeighboursState
        self._controller.setUp(initialNeighboursState = initialNeighboursState,
                               initialAgentState =  self._initialState,
                               initializationTime     = initializationTime,
                               allowSlackSatisfaction=allowSlackSatisfaction) 
        self._hasTasks = len(self._controller._StlTasks)
        
    
    def senseCollision(self,otherAgentsState : list) :
        """ Checks if there is a collsion"""
        self._possibleCollisionAgents = []
        for agentState in otherAgentsState :
            if np.linalg.norm(agentState - self._currentState) <= self._controller._sensingRadius :
                self._possibleCollisionAgents  += [agentState]
        
        if len(self._possibleCollisionAgents)>= self._controller._maxNumCollisionChecks :
            raise RuntimeError("More possible collsions than what the agent can handle are witnessed. please review the simulation and add possible collsion checks when necessay")
        
    def receiveNeighbourState(self,commNeighboursState : dict[int,np.ndarray]) :
        
        self._currentCommNeighboursState = commNeighboursState
        self._neighboursStatesUpdated = True # now you have updated information
           
        
    def printCurrentConstraintValue(self): 
        print("--------------------------------------------------------------------------------------------------")
        print(f"List of constraints evaluation for agent {self._agentID}")
        print("--------------------------------------------------------------------------------------------------")
        if len(self._controller._barrierConstraints) != 0 :
            for kk,constraint in enumerate(self._controller._barrierConstraints) :
                print(f"constraint {kk} value :{self._controller._opti.debug.value(constraint)}")
            print(f"time :{self._controller._opti.debug.value(self._controller._tPar)}")
            print(f"state :{self._controller._opti.debug.value(self._controller._currentAgentStatePar)}")
        
        else :
            print(f"Agent {self._agentID} does not have any constraint")
    
    def printCurrentBarrierValue(self) :
        print("--------------------------------------------------------------------------------------------------")
        print(f"List of barriers value for agent {self._agentID}")
        print("--------------------------------------------------------------------------------------------------")
        if len(self._controller._barrierValues) != 0 :
            for kk,barrier in enumerate(self._controller._barrierValues) :
                print(f"constraint {kk} value :{self._controller._opti.debug.value(barrier)}")   
                
            print(f"time :{self._controller._opti.debug.value(self._controller._tPar)}")
            print(f"state :{self._controller._opti.debug.value(self._controller._currentAgentStatePar)}")
        
        else :
            print(f"Agent {self._agentID} does not have any constraint")
    
    
    def step(self,time)  :
        
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if (not self._neighboursStatesUpdated) and self._hasTasks :
            raise NotImplementedError("you did not update the state of the neighbours")
        
        try :
            self._currentControlInput    = self._controller.computeControlInput(currentCommNeighboursState =  self._currentCommNeighboursState,currentAgentState= self._currentState,time=time,collidingNeighbours=self._possibleCollisionAgents)
        except Exception as error:
            print("-------------------------------------------------------------------------")
            print(f"AGENT {self._agentID} FAILED TO FIND A CONTROL INPUT TO THE OPTIMIZATION")
            print("THE FOLLOWING ERROR OCCURRED")
            print(error)
            print("-------------------------------------------------------------------------")
            print(self._controller._opti.debug.show_infeasibilities())
            exit()
            
            
            
            
        self._currentState           = self._dynamics(self._currentState,self._currentControlInput)
        self._neighboursStatesUpdated = False # so at the next step it is required that you update the state of the neighbours
        
        return self._currentState
       
       
    def getListOfBarrierValuesAtCurrentTime(self)  :
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if len(self._controller._barrierValues) != 0 :
            values = []
            for barrier in self._controller._barrierValues :
                values += [self._controller._opti.debug.value(barrier)]
            return values
        else :
            return []        
        
    def getListOfBarrierConstraintValuesAtCurrentTime(self)  :
        if not self._controller._previouslyInitalised :
            raise NotImplementedError("controller for this agent was not initialised")
        if len(self._controller._barrierConstraints) != 0 :
            values = []
            for constraint in self._controller._barrierConstraints :
                values += [self._controller._opti.debug.value(constraint)]
            return values
        else :
            return []  