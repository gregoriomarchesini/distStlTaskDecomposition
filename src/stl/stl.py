import numpy as np
import casadi as ca
import polytope as pc
from   typing import TypeAlias, Union
from   abc import ABC, abstractmethod
import sys
import matplotlib.pyplot as plt

from loggers.loggers import get_logger

UniqueIdentifier : TypeAlias = int #Identifier of a single agent in the system
SmoothMinBarrier         = Union["IndependentSmoothMinBarrierFunction", "CollaborativeLinearBarrierFunction"]
LinearBarrier            = Union["IndependentLinearBarrierFunction", "CollaborativeLinearBarrierFunction"]
CollaborativeBarrierType = Union["CollaborativeLinearBarrierFunction","CollaborativeSmoothMinBarrierFunction"]
IndependentBarrierType   = Union["IndependentLinearBarrierFunction","IndependentSmoothMinBarrierFunction"]

# Temporal Operators 
class TimeInterval :
    """ Time interval class to represent time intervals in the STL tasks"""
    def __init__(self,a:float|None = None,b:float|None =None) -> None:
        
        
        if any([a==None,b==None]) :
            if (not all(([a==None,b==None]))) :
                raise ValueError("Only empty set is allowed to have None Values for both the extreems a and b of the interval. Please revise your input")
            elif  all(([a==None,b==None])) : # empty set
                self._a = a
                self._b = b
        else :    
            try :
                self._a = float(a)
                self._b = float(b)
            except :
                raise ValueError(f"The given time instants must be convertible to float. Given types are {type(a)}  and {type(b)}")
            
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
            
        
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
    
    
    def __contains__(self,time: Union[float,"TimeInterval"]) -> bool:
        "returns true if a given time instant of time interval is contained in the interval"
        
        if isinstance(time,TimeInterval) :
            
            a1,b1 = time.a,time.b
            a2,b2 = self._a,self._b
            
            if a1>=a2 and b1<=b2 :
                return True
            else :
                return False
        else :
            time = float(time)
            if (self._a >= time) and (time <= self._b) :
                return True
            else :
                return False
            
    def union(self,timeInt:"TimeInterval") -> "TimeInterval":
        """returns the union of two intervals if the two are intersecting"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.is_empty() :
            return timeInt.getCopy()
        if timeInt.is_empty() :
            return self.getCopy()
        
        if not self.__truediv__(timeInt).is_empty() :
            return TimeInterval(a = min(a1,a2), b = max(b1,b2))
        else :
            raise ValueError("The two intervals are not intersecting. The union is not defined by a single interval")
        
    def can_be_merged_with(self,timeInt:"TimeInterval") -> bool:
        """returns true if the two intervals can be merged into a single interval because the two intervals are intersecting"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if self.is_empty() :
            return True
        if timeInt.is_empty() :
            return True
        
        if not self.__truediv__(timeInt).is_empty() :
            return True
        else :
            return False
    
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
    
    def __matmul__(self, predicate : "PolytopicPredicate") -> "StlTask":
        
        return StlTask(temporal_operator = self, predicate = predicate)

class G(TemporalOperator):
    def __init__(self,a:float,b:float) -> None:
        
        try :
            time_interval = TimeInterval(a,b)
        except Exception as e:
            raise ValueError("There was error constructing the temporal operator. The raised exception is : " + str(e))
        
        super().__init__(time_interval)
        self._time_of_satisfaction   : float  = self._time_interval.a
        self._time_of_remotion      : float   = self._time_interval.b
    
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
    
    def __str__(self):
        return f"G_[{self._time_interval.a}, {self._time_interval.b}]"
    
    
    
class F(TemporalOperator):
    def __init__(self,a:float, b:float) -> None:
        """ Eventually operator
        Args:
            time_interval (TimeInterval): time interval that referes to the eventually operator
            time_of_satisfaction (float): time at which the eventually operator is satisfied (assigned randomly if not specified)

        Raises:
            ValueError: if time of satisfaction is outside the time interval range
        """
        try :
            time_interval = TimeInterval(a,b)
        except Exception as e:
            raise ValueError("There was error constructing the temporal operator. The raised exception is : " + str(e))
        super().__init__(time_interval)
        self._time_of_satisfaction = time_interval.a + np.random.rand()*(time_interval.b- time_interval.a)
        self._time_of_remotion     = self._time_of_satisfaction
        
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
    
    def __str__(self):
        return f"F_[{self._time_interval.a}, {self._time_interval.b}]"
    



class PolytopicPredicate(ABC):
    """Abstract class to define polytopic predicates"""
    
    def __init__(self, polytope_0: pc.Polytope , 
                       center:     np.ndarray) -> None:
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
            self._vertices    = [np.expand_dims(vertex,1) for vertex in  [*pc.extreme(self._polytope)] ] # unpacks vertices as a list of column s
            self._num_vertices = len(self._vertices) 
        except:
            raise RuntimeError("There was an error in the computation of the vertices for the polytope. Make sure that your polytope is closed since this is the main source of failure for the algorithm")

    @property
    def state_space_dim(self)-> int:
        return self._state_space_dim
    @property
    def vertices(self) -> list[np.ndarray]:
        return self._vertices
    @property
    def num_vertices(self):
        return self._num_vertices
    @property
    def polytope(self):
        return self._polytope
    @property
    def center(self):
        if self._is_parametric :
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
    
    @property
    @abstractmethod
    def contributing_agents(self):
        pass
    
    def plot(self):
        """Plot the polytope"""
        self._polytope.plot()
    
            
    
class IndependentPredicate(PolytopicPredicate):
    def __init__(self,polytope_0: pc.Polytope , 
                      agent_id  : UniqueIdentifier,
                      center    : np.ndarray) -> None:
        
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
    
    @classmethod
    def create_parametric_predicate(cls, polytope_0:pc.Polytope, agent_id: UniqueIdentifier) -> "IndependentPredicate":
        return cls(polytope_0,agent_id,np.empty((0)))
    
        
class CollaborativePredicate(PolytopicPredicate):
    def __init__(self,polytope_0     : pc.Polytope , 
                      source_agent_id: UniqueIdentifier,
                      target_agent_id: UniqueIdentifier,
                      center         : np.ndarray) -> None:
        
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
    
    @classmethod
    def create_parametric_predicate(cls,polytope_0     : pc.Polytope , source_agent_id: UniqueIdentifier, target_agent_id: UniqueIdentifier,) -> "IndependentPredicate":
        return cls(polytope_0,source_agent_id,target_agent_id, np.empty((0)))
        

class StlTask:
    """STL TASK"""
    
    _id_generator = 0 # counts instances of the class (used to generate unique ids for the tasks).
    def __init__(self,temporal_operator:TemporalOperator, predicate:PolytopicPredicate):
        
        """
        Args:
            temporal_operator (TemporalOperator): temporal operator of the task (includes time interval)
            predicate (PolytopicPredicate): predicate of the task
        """
        
        # if a predicate function is not assigned, it is considered that the predicate is parametric
        
        if not isinstance(temporal_operator,TemporalOperator) :
            raise ValueError("The temporal operator must be an instance of the TemporalOperator class. Given type is " + str(type(temporal_operator)))
        if not isinstance(predicate,PolytopicPredicate) :
            raise ValueError("The predicate must be an instance of the PolytopicPredicate class. Given type is " + str(type(predicate)))
        
        self._predicate              :PolytopicPredicate  = predicate
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
    
    predicate = CollaborativePredicate.create_parametric_predicate(polytope_0      = polytope , 
                                                                   source_agent_id = source_agent_id,
                                                                   target_agent_id = target_agent_id)
    
    child_task : StlTask = StlTask(temporal_operator = temporal_operator, predicate = predicate)
    return child_task



def get_M_and_Z_matrices_from_inclusion(P_including:StlTask|PolytopicPredicate, P_included:StlTask|PolytopicPredicate) -> tuple[np.ndarray,np.ndarray]:
    
    if isinstance(P_including,StlTask) :
        P_including : PolytopicPredicate = P_including.predicate
    if isinstance(P_included,StlTask) :
        P_included : PolytopicPredicate = P_included.predicate
    
    if P_including.state_space_dim != P_included.state_space_dim :
        raise ValueError("The state space dimensions of the two predicates do not match. Please provide predicates with same state space dimensions")
    
    vertices        = P_included.vertices
    num_vertices    = P_included.num_vertices
    state_space_dim = P_included.state_space_dim # same for both predicates
    
    center          = P_including.center
    eta             = np.vstack((center ,np.array([[1]])))
    
    M = []
    for vertex in vertices:
        G_k = np.hstack((np.eye(state_space_dim),vertex))
        M.append(P_including.polytope.A@ G_k)
        
    M     = np.vstack(M)
    z     = np.expand_dims(P_including.polytope.b,axis=1) # make column
    A_bar = np.hstack((P_including.polytope.A, z))
    Z     = np.kron(np.ones((num_vertices,1)),A_bar)@eta  
    
    return M,Z



def communication_consistency_matrices_for(task:StlTask) -> list[np.ndarray]:
    
    vertices : list[np.ndarray] = task.predicate.vertices
    # assume the first `pos_dim` dimensions are the position dimensions
    S = np.eye(vertices[0].size) 
    
    N = []
    for vertex in vertices:
        Gk = np.hstack((np.eye(task.predicate.state_space_dim),vertex))
        Nk = (S@Gk).T @ (S@Gk)
        N += [Nk]
    
    return  N


def random_2D_polytope(number_hyperplanes : int, max_distance_from_center: float) -> PolytopicPredicate:
    
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
    
    
    return pc.Polytope(A,z)
    


def regular_2D_polytope(number_hyperplanes : int, distance_from_center: float) -> PolytopicPredicate :
    
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
    
    return pc.Polytope(A,z)



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
        

class GammaFunction():
    def __init__(self, gamma_0:float , time_flattening: float, t_0 :float  ) -> None:
        """ gamma_0 + gamma_0/ (t_0 - t_flattening) (t-t_0) : -> linear decaying time function"""
        
        if time_flattening<=t_0 :
            raise ValueError("The time flattening parameter must be greater than the initial time")
        if gamma_0<0 :
            raise ValueError("The initial value of the gamma function must be positive")
        
        self._t0              = float(t_0)
        self._time_flattening = float(time_flattening)
        time_var = ca.MX.sym("time",1)
        self._function = ca.Function("gamma_function",[time_var ],[ ca.if_else(time_var<= self._time_flattening,gamma_0 + gamma_0/(self._t0 - self._time_flattening)*(time_var - self._t0),0)],["time"],["value"])
        self._gradient = ca.Function("gamma_gradient",[time_var],[ca.jacobian(self._function(time_var),time_var)])
        
    def compute(self,x : float| ca.MX | ca.SX | ca.DM=0) -> float| ca.MX | ca.SX | ca.DM:
        if isinstance(x,float) :
            return float(self._function(x))
        else :
            return self._function(x)
    
    def compute_gradient(self,x : float| ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        if isinstance(x,float) :
            return float(self._gradient(x))
        else :
            return self._gradient(x)
    
    def plot(self):
        ax = plt.gca()
        time = np.linspace(self._t0,self._time_flattening*1.20,100)
        values = [self.compute(t) for t in time]
        ax.plot(time,values)
        
    @classmethod
    def flat_gamma(cls):
        return cls(gamma_0=0.,time_flattening=1.,t_0=0.) # create a flat gamma function that is always zero



class SwitchOffFunction():
    def __init__(self,switching_time:float = float("inf")) -> None:
        """_summary_

        Args:
            switching_time (float): _description_
        """        
        
        self._switching_time = float(switching_time)
        time_var = ca.MX.sym("time")
        self._function = ca.Function("switch_function",[time_var ],[ca.if_else(time_var<=self._switching_time,1.,0.)],["time"],["value"])
    
    def compute(self,x : float| ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        if isinstance(x,float) :
            return float(self._function(x))
        else :
            return self._function(x)
    
    def plot(self):
        ax = plt.gca()
        time = np.linspace(0,self._switching_time*1.20,100)
        values = [self.compute(t) for t in time]
        ax.plot(time,values)




class BarrierFunction(ABC):
    def __init__(self) -> None:
        """Create a barrier function in the form   d.T@(x-c) - z + gamma(t) >=0 """
    
    @abstractmethod
    def compute():
        pass
    
    @abstractmethod
    def gradient():
        pass

    @abstractmethod
    def time_derivative_at_time(self,t:float | ca.MX | ca.SX | ca.DM):
        pass
        

class CollaborativeLinearBarrierFunction(BarrierFunction) :
    def __init__(self,d : np.ndarray, c: np.ndarray,z:float, gamma_function: GammaFunction,switch_function:SwitchOffFunction, source_agent:UniqueIdentifier, target_agent:UniqueIdentifier) -> None:
        """Creates a linear barrier function in the form  a direction vector d, a center c and a constant z.
           Given that the user want to enforce the constraint d.T@((x_target- x_source) -c) - z <=0 within a  
           certain time interval, the barrier function is defined as  -d.T@((x_target- x_source) -c) + z + gamma(t) >=0
        """
        
    
        super().__init__()
        self._d = d.flatten()[:,np.newaxis]
        self._c = c.flatten()[:,np.newaxis]
        self._z = float(z)
        self._gamma_function = gamma_function
        self._switch_function = switch_function
        
        
        self._source_agent = source_agent
        self._target_agent = target_agent
        
    @property
    def edge(self):
        return sorted((self._target_agent,self._source_agent))
            
    def flip(self):
        """Flips the direction of the predicate"""
        dummy = self._target_agent
        self._target_agent = self._source_agent
        self._source_agent = dummy
        self._d = -self._d
        self._c = -self._c
        
    def compute(self,x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                     x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                     t: float | ca.MX | ca.SX | ca.DM) -> ca.MX | ca.SX | ca.DM:
        
        
        if isinstance(x_source , np.ndarray) :
            x_source = x_source.flatten()[:,np.newaxis]
        if isinstance(x_target , np.ndarray) :
            x_target = x_target.flatten()[:,np.newaxis]
        
        
        try :
            edge_vector = x_target - x_source
        except :
            raise ValueError(f"The input states should be numpy arrays or casadi objects with the sane number of elements. Given types are " + {x_source} + " and " + {x_target})
        
        barrier  = ( -self._d.T@(edge_vector - self._c) + self._z + self._gamma_function.compute(t)) *self._switch_function.compute(t)
        return barrier
    
    def gradient(self,agent_id: UniqueIdentifier,
                      x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                      x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                      t: float | ca.MX | ca.SX | ca.DM) -> np.ndarray:
        
        if agent_id == self._source_agent :
            return self._d*self._switch_function.compute(t)
        elif agent_id == self._target_agent :
            return -self._d*self._switch_function.compute(t)
        else :
            raise ValueError(f"The agent_id {agent_id} is not part of the agents of the predicate")   
    
    def time_derivative_at_time(self, x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                                      x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                                      t       : float | ca.MX | ca.SX | ca.DM) :
        if isinstance(t,float) :
            t = float(t)
            return float(self._gamma_function.compute_gradient(t)*self._switch_function.compute(t)) # the time derivative of the barrier function is the time derivative of the gamma function
        else :
            return self._gamma_function.compute_gradient(t)*self._switch_function.compute(t)
        
class IndependentLinearBarrierFunction(BarrierFunction):
    
    
    def __init__(self,d : np.ndarray, c: np.ndarray,z:float, gamma_function: GammaFunction,switch_function:SwitchOffFunction, agent_id:UniqueIdentifier) -> None:
        """Create a barrier function in the form   d.T@((x_target- x_source) -c) - z + gamma(t) >=0 """
        super().__init__()
        self._d = d.flatten()[:,np.newaxis]
        self._c = c.flatten()[:,np.newaxis]
        self._z = z
        self._gamma_function = gamma_function
        self._switch_function = switch_function
        self._agent_id = agent_id
    
    

    def compute(self,x: np.ndarray | ca.MX | ca.SX | ca.DM, 
                     t: float | ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        
        if isinstance(x, np.ndarray) :
            x = x.flatten()[:,np.newaxis]
        
        barrier  = ( -self._d.T@(x - self._c) + self._z + self._gamma_function.compute(t))*self._switch_function.compute(t)
        return barrier
    
    def gradient(self, x: np.ndarray | ca.MX | ca.SX | ca.DM, t:float | ca.MX | ca.SX | ca.DM) -> np.ndarray:
        return -self._d.T*self._switch_function.compute(t)
    
    def time_derivative_at_time(self,x: np.ndarray | ca.MX | ca.SX | ca.DM, t:float | ca.MX | ca.SX | ca.DM):
        return self._gamma_function.compute_gradient(t)*self._switch_function.compute(t) # the time derivative of the barrier function is the time derivative of the gamma function


class IndependentSmoothMinBarrierFunction(BarrierFunction):
    def __init__(self, list_of_barrier_functions: list[IndependentLinearBarrierFunction],eta:float = 1) -> None:
        self._list_of_barrier_functions = list_of_barrier_functions
        if eta<= 0 :
            raise ValueError("The smoothing parameter eta must be a positive number")
        
        for barrier_function in list_of_barrier_functions :
            if not isinstance(barrier_function,IndependentLinearBarrierFunction) :
                raise ValueError(f"The input list of barrier functions should of type {IndependentLinearBarrierFunction.__name__}. Found input of type {type(barrier_function)}")
            
        self._eta = eta
        self._smooth_min, self._gradient, self._time_derivative = self._create_smooth_min_function()
        
    
    def _create_smooth_min_function(self):
        
        
        sum = 0
        x = ca.MX.sym("x",2) # assume that the state dimension is 2 for now 
        t = ca.MX.sym("t")
        big_number = 1E2
        
        cumulative_switch = 0
        for barrier_function in self._list_of_barrier_functions :
            cumulative_switch += barrier_function._switch_function.compute(t) # to set the barrier at zero when all the barriers have been switched off
            sum += ca.exp(-self._eta*barrier_function.compute(x,t) + (1.-barrier_function._switch_function.compute(t))*big_number/self._eta)  
            # ^^ each terms becomes very big when the switch function is off and the gradient of that component will be zero because the linear barrier component sets the gradient to zero (using the switch function)
        
        smooth_min_sym      = -ca.log(sum)/self._eta
        smooth_min_fun      = ca.Function("smooth_min",[x,t],[smooth_min_sym * ca.if_else(cumulative_switch>0. ,1.0,0.)])
        gradient_fun        = ca.Function("smooth_min_gradient",[x,t],[ca.jacobian(smooth_min_sym,x)])
        time_derivative_fun = ca.Function("smooth_min_time_derivative",[x,t],[ca.jacobian(smooth_min_sym,t)])
        
        return smooth_min_fun, gradient_fun, time_derivative_fun
    
    def compute(self,x: np.ndarray | ca.MX | ca.SX | ca.DM, t: float | ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        return self._smooth_min(x,t)

    def gradient(self, x: np.ndarray | ca.MX | ca.SX | ca.DM, t: float | ca.MX | ca.SX | ca.DM) -> np.ndarray| ca.MX | ca.SX | ca.DM:
        return self._gradient(x,t)
    
    def time_derivative_at_time(self,  x: np.ndarray | ca.MX | ca.SX | ca.DM, t: float | ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        return self._time_derivative(x,t)


class CollaborativeSmoothMinBarrierFunction(BarrierFunction):
    def __init__(self, list_of_barrier_functions: list[CollaborativeLinearBarrierFunction],eta:float = 1) -> None:
        
        self._list_of_barrier_functions = list_of_barrier_functions
        barrier_0 = list_of_barrier_functions[0]
        edge      = barrier_0.edge
        
        for barrier_function in list_of_barrier_functions :
            if not isinstance(barrier_function,CollaborativeLinearBarrierFunction) :
                raise ValueError(f"The input list of barrier functions should of type {CollaborativeLinearBarrierFunction.__name__}. Found input of type {type(barrier_function)}")
            
            if barrier_function.edge != edge :
                raise ValueError("The input list of barrier functions should all be defined over the same edge")
        
            elif barrier_function._source_agent != barrier_0._source_agent :
                barrier_function.flip() # flip it so all the function are defined over the same direction
                
        if eta<= 0 :
            raise ValueError("The smoothing parameter eta must be a positive number")
        self._eta = eta
        self._smooth_min, self._gradient, self._time_derivative = self._create_smooth_min_function()
        self._source_agent = barrier_0._source_agent
        self._target_agent = barrier_0._target_agent
    
    @property
    def edge(self):
        return sorted((self._source_agent,self._target_agent))
    @property
    def source_agent(self):
        return self._source_agent
    @property
    def target_agent(self):
        return self._target_agent
    
    
    def _create_smooth_min_function(self):
        
        sum = 0
        x_source   = ca.MX.sym("x",2) # assume that the state dimension is 2 for now 
        x_target   = ca.MX.sym("x",2) # assume that the state dimension is 2 for now
        t          = ca.MX.sym("t")
        
        big_number = 1E2
        cumulative_switch = 0
        for barrier_function in self._list_of_barrier_functions :
            cumulative_switch += barrier_function._switch_function.compute(t) # to set the barrier at zero when all the barriers have been switched off
            sum += ca.exp(-self._eta* ( barrier_function.compute(x_source=x_source,x_target=x_target,t=t)  + (1.-barrier_function._switch_function.compute(t))*big_number/self._eta) ) 
            # ^^ each terms becomes very big when the switch function is off and the gradient of that component will be zero because the linear barrier component sets the gradient to zero (using the switch function)
        
        smooth_min_sym      = -1/self._eta * ca.log(sum)
        smooth_min_fun      = ca.Function("smooth_min",[x_source,x_target,t],[smooth_min_sym* ca.if_else(cumulative_switch>0. ,1.0,0.)])
        gradient_fun        = ca.Function("smooth_min_gradient",[x_source,x_target,t],[ca.jacobian(smooth_min_sym,x_target)]) 
        time_derivative_fun = ca.Function("smooth_min_time_derivative",[x_source,x_target,t],[ca.jacobian(smooth_min_sym,t)])
        
        return smooth_min_fun, gradient_fun, time_derivative_fun
    
    def compute(self,x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                     x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                     t: float | ca.MX | ca.SX | ca.DM) -> float| ca.MX | ca.SX | ca.DM:
        
        
        if isinstance(x_source , np.ndarray) :
            x_source = x_source.flatten()[:,np.newaxis]
        if isinstance(x_target , np.ndarray) :
            x_target = x_target.flatten()[:,np.newaxis]
        
        return self._smooth_min(x_source,x_target,t)
    
    def gradient(self,agent_id: UniqueIdentifier,
                      x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                      x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                      t: float | ca.MX | ca.SX | ca.DM) -> np.ndarray:
        
        if agent_id == self._source_agent :
            return - self._gradient(x_source,x_target,t)
        elif agent_id == self._target_agent :
            return self._gradient(x_source,x_target,t)
        else :
            raise ValueError(f"The agent_id {agent_id} is not part of the agents of the predicate")   
    
    def time_derivative_at_time(self,x_source: np.ndarray | ca.MX | ca.SX | ca.DM, 
                                     x_target: np.ndarray | ca.MX | ca.SX | ca.DM, 
                                     t: float | ca.MX | ca.SX | ca.DM):
        return self._time_derivative(x_source,x_target,t) # the time derivative of the barrier function is the time derivative of the gamma function
    

def create_linear_barriers_from_task(task : StlTask, initial_conditions : dict[UniqueIdentifier,np.ndarray], t_init : float = 0 , maximum_control_input_norm : float = 1E5) -> list[BarrierFunction]:
    """
    Create Barrier Functions from a given STL task
    """
    
    # Heuristic on maximum time decay :
    # linear barrier constraint   alpha_coeff*(-a.T@x + z) +   -a.T@u + gamma_dot(t) >=0
    # The barrier inside the safe set is positive while gamma_dot is always negative.
    # So in the worse case scenario alpha_coeff*(-a.T@x + z)=0 (a.k.a border of safe set).
    # So now we have -a.T@u + gamma_dot(t) >=0  
    # with the best input possible we have |a||u|_max <= |gamma_dot(t)|
    # 
    
    initial_conditions = make_initial_conditions_a_column(initial_conditions)
    # get task specifics
    contributing_agents  = task.predicate.contributing_agents # list of contributing agents. In the case of this controller this is always going to be 2 agents : the self agent and another agent
    
    # check that all the agents are present
    if not all([agent_id in initial_conditions.keys() for agent_id in contributing_agents]) :
        raise ValueError("The initial conditions for the contributing agents are not complete. Contributing agents are " + str(contributing_agents) + " and the initial conditions are given for " + str(initial_conditions.keys()))
    

    # get the initial value of the barrier function
    if isinstance(task.predicate,IndependentPredicate) :
        predicate = task.predicate
        initial_state = initial_conditions[predicate.agent_id]
        barrier_value_at_zero = predicate.polytope.b[:,np.newaxis] - predicate.polytope.A @ (initial_state - predicate.center) # b - A*(x_0-center) >=0 
    
    elif isinstance(task.predicate,CollaborativePredicate) :
        predicate = task.predicate
        initial_state = initial_conditions[predicate.target_agent] - initial_conditions[predicate.source_agent]  # The initial state is the initial state of the edge in this case
        barrier_value_at_zero = - predicate.polytope.A @ (initial_state - predicate.center) + predicate.polytope.b[:,np.newaxis]    # b - A*(x_0-center) >=0 
    
    # extract temporal properties 
    time_of_satisfaction = task.temporal_operator.time_of_satisfaction
    time_of_remotion     = task.temporal_operator.time_of_remotion
    switch_function      = SwitchOffFunction(switching_time=time_of_remotion)
    barriers = []

    # create a list of gamma functions for each hyperplane
    for jj in range(predicate.num_hyperplanes) :
        
        maximum_heuristic_allowed_time_decay = np.linalg.norm(predicate.polytope.A[jj,:])*maximum_control_input_norm
        gamma_0_max = maximum_heuristic_allowed_time_decay*(time_of_satisfaction -t_init) # given by max speed
        
        if barrier_value_at_zero[jj] < 0. : # you are outside the super level set of the predicate
            gamma_0_min = -barrier_value_at_zero[jj] # given by the fact that you need to set the initial condition at the border of the set
            
            if time_of_satisfaction<=t_init :
               raise ValueError(("The input task has a time_of_satisfaction value that is lower or equal than the initial time but the agent initial condition" + 
                                 "is outside the predicate super level set of the task. This implies that the task cannot be satisfied from the given t_init value."+
                                 "This can happen for example if the input task has a temporal operator G_[a,b] and t_init<=a"))
            
            if gamma_0_max <= gamma_0_min :
                raise RuntimeError((f"From the provided maximum control input it seems that this barrier will not be satisfied with the given maximum control input." + 
                                    "Probably the initial condition is too far from the zero-super-level set of the predicate assigned to the task. Either select better" +
                                    "initial conditions or increase the maximum control input"))
            
            gamma_0   =  gamma_0_min  + 0.02*(gamma_0_max - gamma_0_min) # 80% of the maximum value 
            gamma_fun = GammaFunction(gamma_0         = gamma_0,
                                      time_flattening = time_of_satisfaction,
                                      t_0             = t_init)
        
        
        
        elif barrier_value_at_zero[jj] == 0. : # you are the border of the super level set of the predicate
            if time_of_satisfaction<t_init :
               raise ValueError("The input task has a time_of_satisfaction value that is lower than the initial time but the agent initial condition at the border of the super level set of the task. This implies that the task cannot be satisfied from the given t_init value. This can happen for example if the input task has a temporal operator G_[a,b] and t_init<=a")
            
            elif time_of_satisfaction == t_init :
                gamma_fun = GammaFunction.flat_gamma()
            else :
                heuristic_ratio = 0.1
                gamma_fun = GammaFunction(gamma_0         = gamma_0_max*heuristic_ratio,
                                          time_flattening = time_of_satisfaction,
                                          t_0             = t_init)
                
            
        else : # you are inside the super level set of the predicate
           # In general it is better not have to have to many tasks with this condition but rather set som trigger times where you filter out the tasks that where already satisfied 
            
            if time_of_satisfaction<=t_init :
                gamma_fun = GammaFunction.flat_gamma()
            else :
                heuristic_ratio = 0.1
                gamma_fun       = GammaFunction(gamma_0         = gamma_0_max*heuristic_ratio,
                                                time_flattening = time_of_satisfaction,
                                                t_0             = t_init)
    
        if isinstance(task.predicate,IndependentPredicate) :
            barrier = IndependentLinearBarrierFunction(d = predicate.polytope.A[jj,:],
                                                       c = predicate.center,
                                                       z = predicate.polytope.b[jj],
                                                       gamma_function = gamma_fun,
                                                       agent_id = predicate.agent_id,
                                                       switch_function = switch_function)
    
        elif isinstance(task.predicate,CollaborativePredicate) :
            barrier = CollaborativeLinearBarrierFunction(d = predicate.polytope.A[jj,:],
                                                         c = predicate.center,
                                                         z = predicate.polytope.b[jj],
                                                         gamma_function  = gamma_fun,
                                                         source_agent    = predicate.source_agent,
                                                         target_agent    = predicate.target_agent,
                                                         switch_function = switch_function)
        barriers += [barrier]
        
    return barriers
        


def make_initial_conditions_a_column(initial_conditions:dict[UniqueIdentifier,np.ndarray]) -> dict[UniqueIdentifier,np.ndarray]:
    """Make sure that the initial conditions are column vectors"""
    
    for agent_id in initial_conditions.keys() :
        if initial_conditions[agent_id].ndim == 1 :
            initial_conditions[agent_id] = np.expand_dims(initial_conditions[agent_id],1)
    
    return initial_conditions


#todo : clean up the funciton here
def plot_contour(smooth_min :CollaborativeBarrierType | IndependentBarrierType  ,ax : None,t):
    x = np.linspace(-8,8,100)
    y = np.linspace(-8,8,100)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]) :
        for j in range(X.shape[1]) :
            try :
                Z[i,j] = smooth_min.compute(x_target=np.array([[X[i,j]],[Y[i,j]]]),
                                            x_source=np.zeros((2,1)) ,
                                            t=t)
            except :
                raise NotImplementedError("plotting only allowed for 2D states")
    return ax.contourf(X,Y,Z,cmap=plt.cm.bone)


def filter_tasks_by_time_limit(tasks:list[StlTask], initial_time: float, final_time :float) -> list[StlTask]:
    return [task for task in tasks if task.temporal_operator.time_of_satisfaction >= initial_time and task.temporal_operator.time_of_satisfaction <= final_time]



if __name__ == "__main__" :
    
    timeint1 = TimeInterval(0,10)
    timeint2 = TimeInterval(10,20)
    #intersection 
    print(timeint1.can_be_merged_with(timeint2))
    print(timeint1.union(TimeInterval(None,None)))
    