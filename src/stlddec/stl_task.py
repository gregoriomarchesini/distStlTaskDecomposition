import numpy as np
import casadi as ca
import polytope as pc
from   typing import TypeAlias
from   abc import ABC, abstractmethod

UniqueIdentifier : TypeAlias = int #Identifier of a single agent in the system

# Temporal Operators 
class TimeInterval :
    """ Time interval class to represent time intervals in the STL tasks"""
    def __init__(self,a:float|None = None,b:float|None =None) -> None:
        
        
        if any([a==None,b==None]) and (not all(([a==None,b==None]))) :
            raise ValueError("only empty set is allowed to have None Values for both the extreems a and b of the interval. Please revise your input")
        elif  any([a==None,b==None]) and (all(([a==None,b==None]))) : # empty set
            self._a = a
            self._b = b
        else :    
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
         
            self._a = float(a)
            self._b = float(b)
        
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b
    
    @property
    def measure(self) :
        if self.is_empty() :
            return None # empty set has measure None
        return self._b - self._a
    
    @property
    def aslist(self) :
        return [self._a,self._b]
    
    def is_empty(self)-> bool :
        if (self._a == None) and (self._b == None) :
            return True
        else :
            return False
        
    def is_singular(self)->bool:
        a,b = self._a,self._b
        if a==b :
            return True
        else :
            return False
        
        
        
    def __truediv__(self,timeInt:"TimeInterval") -> "TimeInterval" :
        """returns interval Intersection"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        
        # If any anyone is empty return the empty set.
        if self.is_empty() or timeInt.is_empty() :
            return TimeInterval(a = None, b = None)
        
        # check for intersection.
        if b2<a1 :
            return TimeInterval(a = None, b = None)
        elif a2>b1 :
            return TimeInterval(a = None, b = None)
        else :
            return TimeInterval(a = max(a2,a1), b = min(b1,b2))
            
        
    
    def __eq__(self,timeInt:"TimeInterval") -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
    
    def __ne__(self,timeInt:"TimeInterval") -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
        
    
    def __lt__(self,timeInt:"TimeInterval") -> "TimeInterval":
        """strict subset relations self included in timeInt ::: "TimeInterval" < timeInt """
        
        intersection = self.__truediv__(timeInt)
        
        if intersection.is_empty() :
            return False
        
        if (intersection!=self) :
            return False
        
        if (intersection == self) and (intersection != timeInt) :
            return True
        else :
            return False # case of equality between the sets
    
        
    def __le__(self,timeInt:"TimeInterval") -> "TimeInterval" :
        """subset (with equality) relations self included in timeInt  ::: "TimeInterval" <= timeInt """
        
        intersection = self.__truediv__(timeInt)
        
        if intersection.is_empty() :
            return False
        
        if (intersection!=self) :
            return False
        
        if (intersection == self) :
            return True
   
        
        
    def __str__(self):
        return f"[{self.a},{self.b}]"
    

    def getCopy(self) :
        return TimeInterval(a = self.a, b=self.b)
    

class TemporalOperator(ABC):
    def __init__(self,time_interval:TimeInterval) -> None:
        self._time_interval         : TimeInterval = time_interval
    
    @property
    @abstractmethod
    def time_of_satisfaction(self) -> float:
        pass
    
    @property
    @abstractmethod
    def time_of_remotion(self) -> float:
        pass
    @property
    def time_interval(self) -> TimeInterval:
        return self._time_interval
    

class AlwaysOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval) -> None:
        
        super().__init__(time_interval)
        self._time_of_satisfaction   : float  = self._time_interval.a
        self._time_of_remotion      : float   = self._time_interval.b
    
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion

    
class EventuallyOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval,time_of_satisfaction:float=None) -> None:
        """ Eventually operator
        Args:
            time_interval (TimeInterval): time interval that referes to the eventually operator
            time_of_satisfaction (float): time at which the eventually operator is satisfied (assigned randomly if not specified)

        Raises:
            ValueError: if time of satisfaction is outside the time interval range
        """
        super().__init__(time_interval)
        self._time_of_satisfaction : float       = time_of_satisfaction
        self._time_of_remotion    : float        = self._time_of_satisfaction
        
        if time_of_satisfaction == None : # if not given pick a random number in the interval
            self._time_of_satisfaction = time_interval.a + np.random.rand()*(time_interval.b- time_interval.a)
            self._time_of_remotion     = self._time_of_satisfaction
            
        elif time_of_satisfaction<time_interval.a or time_of_satisfaction>time_interval.b :
            raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{time_interval.a},{time_interval.b}]")
        
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion



#!TODO: Still under development. Do not use for now.
# class AlwaysEventuallyOperator(TemporalOperator):
#     def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
#         #pick it random if not given
#         if eventually_time_of_satisfaction == None :
#             eventually_time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
#         else :
#             if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
#                 raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
        
#         self._period               : TimeInterval = eventually_time_of_satisfaction # from the point "a" of the evetually, we have that the task is satisfied everty evetually_time_of_satsifaction
        
#         self._time_of_satisfaction : float       = always_time_interval.a + eventually_time_interval.a # we satisfy at the initial time of the always first
        
#         self._time_of_remotion     : float       = self._time_of_satisfaction +  np.ceil((always_time_interval.b - self._time_of_satisfaction)/ self._period) * self._period

        
#     @property
#     def time_of_satisfaction(self) -> float:
#         return self._time_of_satisfaction
    
#     @property
#     def time_of_remotion(self) -> float:
#         return self._time_of_remotion
   
#     @property
#     def period(self) -> TimeInterval:
#         return self._period

#!TODO: Still under development. Do not use for now.
# class EventuallyAlwaysOperator(TemporalOperator):
#     def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
        
#         if eventually_time_of_satisfaction == None :
#              self._time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
#         else :
#             if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
#                 raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
            
#         self._time_of_satisfaction : float    = eventually_time_of_satisfaction
#         self._time_of_remotion     : float    = self._time_of_satisfaction  + always_time_interval.period

#     @property
#     def time_of_satisfaction(self) -> float:
#         return self._time_of_satisfaction
    
#     @property
#     def time_of_remotion(self) -> float:
#         return self._time_of_remotion
   
    

# Predicates and tasks

class AbstractPolytopicPredicate(ABC):
    """Abstract class to define polytopic predicates"""
    
    @abstractmethod
    def __init__(self, polytope_0: pc.Polytope , 
                       center:     np.ndarray = np.empty((0))) -> None:
        """_summary_

        Args:
            polytope_0 (pc.Polytope):  zero centered polytope 
            center (np.ndarray):       polytope center
            is_parametric (bool, optional): defines is a predicate has to be considered parametric or not. Defaults to False.

        Raises:
            RuntimeError: _description_
        """
        
        
        # when the predicate is parametric, then the center is assumed to be the one assigned to the orginal predicate from which the predicate is derived for the decomspotion
        
        if center.size == 0 :
            self._is_parametric = True
        else :
            self._is_parametric = False
            
        if center.ndim == 1 and center.size != 0 :
            center = np.expand_dims(center,1) 
        self._center  = center # center of the polygone
            

        self._polytope   = polytope_0.copy() # to stay on the safe side and avoid multiple references to the same object from different predicates.
        
        if not self._polytope.contains(np.zeros((self._polytope.A.shape[1],1))):
            raise ValueError("The polytope should contain the origin to be considered a valid polytope.")
        
        self._num_hyperplanes , self._state_space_dim = np.shape(polytope_0.A)
            
        try :
            self._vertices    = [np.expand_dims(vertex,1) for vertex in  [*pc.extreme(self._polytope)] ] # unpacks vertices as a list of column vectors
            self._num_vertices = len(self._vertices) 
        except:
            raise RuntimeError("There was an error in the computation of the vertices for the polytope. Make sure that your polytope is closed since this is the main source of failure for the algorithm")

    @property
    def state_space_dim(self)-> int:
        return self._state_space_dim
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices
    @property
    def num_vertices(self):
        return self._num_vertices
    @property
    def polytope(self):
        return self._polytope
    @property
    def center(self):
        if self._center.size == 0 :
            raise ValueError("The predicate is parametric and does not have a center")
        return self._center
    @property
    def A(self):
        return self._polytope.A
    @property
    def b(self):
        return self._polytope.b
    @property
    def num_hyperplanes(self):
        return self._num_hyperplanes
    @property
    def is_parametric(self) :
        return self._is_parametric
    

class IndependentPredicate(AbstractPolytopicPredicate):
    def __init__(self,polytope_0: pc.Polytope , 
                      agent_id: UniqueIdentifier,
                      center:     np.ndarray = np.empty((0))) -> None:
        
        # initialize parent predicate
        super().__init__(polytope_0 ,center)
        
        self._contributing_agents = [agent_id]
        self._agent_id = agent_id
    
    @property
    def contributing_agents(self):
        return [self._agent_id]
    @property
    def agent_id(self):
        return self._agent_id
    
        
class CollaborativePredicate(AbstractPolytopicPredicate):
    def __init__(self,polytope_0: pc.Polytope , 
                      source_agent_id: UniqueIdentifier,
                      target_agent_id: UniqueIdentifier,
                      center:     np.ndarray = np.empty((0))) -> None:
        
        # initialize parent predicate
        super().__init__(polytope_0,center)
        
        self._source_agent_id = source_agent_id
        self._target_agent_id  = target_agent_id
        if source_agent_id == target_agent_id :
            raise ValueError("The source and target agents must be different since this is a collaborative predictae. Use the IndependentPredicate class for individual predicates")
        
    @property
    def source_agent(self):
        return self._source_agent_id
    @property
    def target_agent(self):
        return self._target_agent_id
    @property
    def contributing_agents(self):
        return [self._source_agent_id,self._target_agent_id]
    
    def flip(self):
        """Flips the direction of the predicate"""
        
        # A @ (x_i-x_j - c) <= b   =>  A @ (e_ij - c) <= b 
        # becomes 
        # A @ (x_j-x_i + c) >= -b   =>  -A @ (e_ji + c) <= b  =>  A_bar @ (e_ji - c_bar) <= b
        
        # swap the source and target
        dummy = self._target_agent_id
        self._target_agent_id = self._source_agent_id
        self._source_agent_id = dummy
        
        # change center direction of the predicate
        if not self._is_parametric :
            self._center = - self._center
        # change matrix A
        self._polytope = pc.Polytope(-self._polytope.A,self._polytope.b)
        

class StlTask:
    """STL TASK"""
    
    _id_generator = 0 # counts instances of the class (used to generate unique ids for the tasks).
    def __init__(self,temporal_operator:TemporalOperator, predicate:AbstractPolytopicPredicate):
        
        """
        Args:
            temporal_operator (TemporalOperator): temporal operator of the task (includes time interval)
            predicate (PolytopicPredicate): predicate of the task
        """
        
        # if a predicate function is not assigned, it is considered that the predicate is parametric
        
        self._predicate              :AbstractPolytopicPredicate  = predicate
        self._temporal_operator      :TemporalOperator    = temporal_operator
        self._task_id                :int                 = StlTask._id_generator #unique id for this task
        
        # spin the id_generator counter.
        StlTask._id_generator += 1 
        
    @property
    def predicate(self):
        return self._predicate
    @property
    def temporal_operator(self):
        return self._temporal_operator
    @property
    def state_space_dimension(self):
        return self._predicate.state_space_dim     
    @property
    def is_parametric(self):
        return self._predicate.is_parametric
    @property
    def predicate(self):
        return self._predicate
    @property
    def task_id(self):
        return self._task_id
    
    
    def flip(self) :
        """Flips the direction of the predicate"""
        if not isinstance(self._predicate,CollaborativePredicate) :
            raise ValueError("The task is not a collaborative task. Individual tasks cannot be flipped")
        self._predicate.flip()
        

def create_parametric_collaborative_task_from(task : StlTask, source_agent_id:UniqueIdentifier, target_agent_id : UniqueIdentifier) -> StlTask :
    """Creates a parametric collaborative task from a given collaborative task, with anew source and target agents"""
    
    if isinstance(task.predicate,IndependentPredicate) :
        raise ValueError("The task is not a collaborative task. Individual tasks are not supported")
    
    polytope          = task.predicate.polytope.copy()
    temporal_operator = task.temporal_operator
    
    predicate =  CollaborativePredicate(polytope_0      = polytope , 
                                        source_agent_id = source_agent_id,
                                        target_agent_id = target_agent_id)
    
    child_task:StlTask = StlTask(temporal_operator = temporal_operator, predicate = predicate)
    return child_task



def get_M_and_Z_matrices_from_inclusion(P_including:StlTask|AbstractPolytopicPredicate, P_included:StlTask|AbstractPolytopicPredicate) -> tuple[np.ndarray,np.ndarray]:
    
    if isinstance(P_including,StlTask) :
        P_including : AbstractPolytopicPredicate = P_including.predicate
    if isinstance(P_included,StlTask) :
        P_included : AbstractPolytopicPredicate = P_included.predicate
    
    if P_including.state_space_dim != P_included.state_space_dim :
        raise ValueError("The state space dimensions of the two predicates do not match. Please provide predicates with same state space dimensions")
    
    vertices        = P_included.vertices
    num_vertices    = P_included.num_vertices
    state_space_dim = P_included.state_space_dim # same for both predicates
    
    
    M = []
    for vertex in vertices:
        G_k = np.hstack((np.eye(state_space_dim),vertex))
        M.append(P_including.polytope.A@ G_k)
        
    M     = np.vstack(M)
    z     = np.expand_dims(P_including.polytope.b,axis=1) # make column
    A_bar = np.hstack((P_including.polytope.A, z))
    Z     = np.kron(np.ones((num_vertices,1)),A_bar)    
    
    return M,Z



def communication_consistency_matrices_for(task:StlTask,pos_dim:int = 2) -> list[np.ndarray]:
    
    vertices = task.predicate.vertices
    # assume the first `pos_dim` dimensions are the position dimensions
    S = np.eye(pos_dim) 
    
    N = []
    for vertex in vertices:
        Gk = np.hstack((np.eye(task.predicate.state_space_dim),vertex))
        Nk = (S@Gk).T @ (S@Gk)
        N += [Nk]
    
    return  N


def random_2D_polytope(number_hyperplanes : int, max_distance_from_center: float) -> AbstractPolytopicPredicate:
    
    number_hyperplanes = int(number_hyperplanes) # convert to int
    
    if max_distance_from_center<=0 :
        raise ValueError("Distance_from_center must be a positive number")
    if number_hyperplanes<=2 :
        raise ValueError("Number of hyperplanes needs to be higher than 2 in two dimensions in order to form a closed polytope (a simplex).")
    
    step = 360/number_hyperplanes
    A = np.zeros((number_hyperplanes,2))
    z = np.random.random((number_hyperplanes,1))*max_distance_from_center
    
    theta = 2*np.pi*np.random.rand()
    R     = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
        random_direction = np.array([[np.sin(angle)],[np.cos(angle)]])
        A[jj,:] = np.squeeze(R@random_direction )
    
    
    return A,z
    


def regular_2D_polytope(number_hyperplanes : int, distance_from_center: float) -> AbstractPolytopicPredicate :
    
    number_hyperplanes = int(number_hyperplanes) # convert to int
    
    if distance_from_center<=0 :
        raise ValueError("Distance_from_center must be a positive number")
    if number_hyperplanes<=2 :
        raise ValueError("Number of hyperplanes needs to be higher than 2 in two dimensions in order to form a closed polytope (a simplex).")
    
    step = 360/number_hyperplanes
    A = np.zeros((number_hyperplanes,2))
    z = np.ones((number_hyperplanes,1))*distance_from_center
    theta = 2*np.pi*np.random.rand()
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
       
       direction = np.array([[np.sin(angle)],[np.cos(angle)]])
       A[jj,:]   = np.squeeze(R@direction)
    
    return A,z



def normal_form(A : np.ndarray,b : np.ndarray) :
    
    # normalize the rows of A
    normA = np.sqrt(np.sum(A**2,axis=1))
    A    = A/normA[:,np.newaxis]
    b    = b/normA
    
    # make sure that the b values are positive
    neg_pos = b<0
    A[neg_pos] = -A[neg_pos]
    b[neg_pos] = -b[neg_pos]
    
    return A,b
        

             
if __name__ =="__main__" :
    
    
    def separate_line():
        print("Passed")
        print("-------------------------------------------------------")
    
    show_figures = False
    
    import matplotlib.pyplot as plt
    # Test 1 : Normal form
    print("Test 1 : Normal form")
    A = np.array([[20,1],[1,-20]])
    b = np.array([[1],[-1]])
    A,b = normal_form(A,b)
    print(A)
    print(np.linalg.norm(A,axis=1))
    
    separate_line()
    # Test 2 : Regular 2D polytope
    print("Test 2 : Regular 2D polytope")
    
    
    dist   = 1
    step   = 2*dist +1
    center = np.array([[0],[0]])
    max_hp = 10
    fig,ax = plt.subplots()
    ax.set_xlim(- step ,max_hp*step+dist)
    ax.set_ylim(- step , step )
    ax.aspect = 'equal'
    
    for hp in range(3,10):
        A,z = regular_2D_polytope(hp,1)

        p = pc.Polytope(A,z+A@center)
        p.plot(alpha=  0.5,ax = ax)
        center = center + np.array([[step],[0]])
    ax.set_title("Regular 2D Polytopes")
    
    separate_line()
    # Test 3 : Random 2D polytope
    print("Test 3 : Random 2D polytope")
    dist   = 1
    step   = 2*dist +1
    center = np.array([[0],[0]])
    max_hp = 10
    fig,ax = plt.subplots()
    ax.set_xlim(- step ,max_hp*step+dist)
    ax.set_ylim(- step , step )
    ax.aspect = 'equal'
    
    for hp in range(3,10):
        A,z = random_2D_polytope(hp,1)

        p = pc.Polytope(A,z+A@center)
        p.plot(alpha=  0.5,ax = ax)
        center = center + np.array([[step],[0]])
    
    ax.set_title("Random 2D Polytopes")
    
    separate_line()
    # Test 4: Test temporal operators 
    print("Test 4: Test temporal operators and time intervals")
    time_interval1 = TimeInterval(0,10)
    time_interval2 = TimeInterval(3,15)
    intersection   = TimeInterval(3,10)
    
    assert time_interval1/time_interval2 == intersection 
    
    time_interval1 = TimeInterval(None,None)
    time_interval2 = TimeInterval(3,15)
    assert time_interval1/time_interval2 == TimeInterval(None,None) 
    assert time_interval1.is_empty()
    
    time_interval1 = TimeInterval(3,3)
    time_interval2 = TimeInterval(3,15)
    assert time_interval1/time_interval2 == TimeInterval(3,3) 
    assert (time_interval1/time_interval1).is_singular()
    separate_line()
    print("Temporal operator G_[0,10]")
    temporal_operator = AlwaysOperator(TimeInterval(0,10))
    print("Always operator time of remotion: ",temporal_operator.time_of_remotion)
    print("Always operator time of satisfaction: ",temporal_operator.time_of_satisfaction)
    separate_line()
    
    temporal_operator = EventuallyOperator(TimeInterval(0,10))
    print("Eventually operator time of remotion: ",temporal_operator.time_of_remotion)
    print("Eventually  operator time of satisfaction: ",temporal_operator.time_of_satisfaction)
    print("Time of satisfaction is the same as time of remotion")
    assert temporal_operator.time_of_satisfaction >= 0
    assert temporal_operator.time_of_satisfaction <= 10
    assert temporal_operator.time_of_remotion == temporal_operator.time_of_satisfaction
    separate_line()
    
    
    # Test 5: Test predicates
    A,b = regular_2D_polytope(5,1)
    P   = CollaborativePredicate(pc.Polytope(A,b.flatten()),source_agent_id=0,target_agent_id=1)    
    # P   = AbstractPolytopicPredicate(pc.Polytope(A,b)) # throws an error
    print(A)
    assert P.is_parametric
    assert P.state_space_dim == 2
    assert P.num_hyperplanes == 5
    assert P.num_vertices == 5
    
    P_ind = IndependentPredicate(pc.Polytope(A,b),agent_id=0)
    assert P_ind.is_parametric
    P_collab = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=0,target_agent_id=1)
    P_collab.flip()
    
    assert P_collab.source_agent == 1
    assert not np.all(np.linalg.norm(P_collab.A +A,axis=1))
    
    separate_line()
    # Test 6: Test tasks
    print("Test 6: Test tasks")
    task = StlTask(AlwaysOperator(TimeInterval(0,10)),P)
    assert task.state_space_dimension == 2
    assert task.is_parametric
    assert task.predicate == P
    assert task.task_id == 0
    
    for jj in range(10):
        task = create_parametric_collaborative_task_from(task,source_agent_id=0,target_agent_id=1)
        assert task.task_id == jj+1
        print("New task created with id: ",task.task_id)
    
    if show_figures:
        plt.show()
    
    # Test inclusions 
    print("Test 7: Test inclusion matrices")
    A,b = regular_2D_polytope(5,1)
    print(b)
    P_including = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=0,target_agent_id=1)
    A,b = regular_2D_polytope(3,1)
    P_included = IndependentPredicate(pc.Polytope(A,b),agent_id=0)
    
    M,Z = get_M_and_Z_matrices_from_inclusion(P_including,P_included)
    print("Matrix M: should be 15x3 and equal to  [A@[I v1] ,A@[I v2],A@[I v3]], with v_i vertices of the included polytope")
    print(M)
    print("Matrix Z: should be 15x3 and equal to  [[A z],[A,z],[A,z]]")
    print(Z)
    