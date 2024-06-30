import numpy as np
import casadi as ca
import polytope as pc
from   typing import TypeAlias
from abc import ABC

from .temporal import TemporalOperator

UniqueIdentifier : TypeAlias = int # identifier of a single agent in the system

# Some support functions    
def first_word_before_underscore(string: str) -> str:
    """split a string by underscores and return the first element"""
    return string.split("_")[0]


def check_barrier_function_input_names(barrier_function: ca.Function)-> bool:
    for name in barrier_function.name_in():
        if not first_word_before_underscore(name) in ["state","time"]:
            return False
    return True    

def check_barrier_function_output_names(barrier_function: ca.Function)->bool:
    for name in barrier_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True

def is_time_state_present(barrier_function: ca.Function) -> bool:
    return "time" in barrier_function.name_in() 


def check_barrier_function_IO_names(barrier_function: ca.Function) -> bool:
    if not check_barrier_function_input_names(barrier_function) :
         raise ValueError("The input names for the predicate functons must be in the form 'state_i' where ''i'' is the agent ID and the output name must be 'value', got input nmaes " + str(function.name_in()) + " and output names " + str(function.name_out()) + " instead")
    
    elif not is_time_state_present(barrier_function) :
        raise ValueError("The time variable is not present in the input names of the barrier function. PLease make sure this is a function of time also (even if time could be not part of the barrier just put it as an input)")
    elif not check_barrier_function_output_names(barrier_function) :
        raise ValueError("The output name of the barrier function must be must be 'value'")
    

def check_predicate_function_input_names(predicate_function: ca.Function)-> bool:
    for name in predicate_function.name_in():
        if not first_word_before_underscore(name) in ["state"]:
            return False
    return True    


def check_predicate_function_output_names(predicate_function: ca.Function)->bool:
    for name in predicate_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True


def check_predicate_function_IO_names(predicate_function: ca.Function) -> bool:
    return check_predicate_function_input_names(predicate_function) and check_predicate_function_output_names(predicate_function)


def state_name_str(agent_id: UniqueIdentifier) -> str:
    """_summary_

    Args:
        agent_id (UniqueIdentifier): _description_

    Returns:
        _type_: _description_
    """    
    return f"state_{agent_id}"

def get_id_from_input_name(input_name: str) -> UniqueIdentifier:
    """Support function to get the id of the agents involvedin the satisfaction of this barrier function

    Args:
        input_names (list[str]): _description_

    Returns:
        list[UniqueIdentifier]: _description_
    """    
    if not isinstance(input_name,str) :
        raise ValueError("The input names must be a string")
    
 
    splitted_input_name = input_name.split("_")
    if 'state' in splitted_input_name :
        ids = int(splitted_input_name[1])
    else :
        raise RuntimeError("The input name must be in the form 'state_i' where ''i'' is the agent ID")
    
    return ids






#################################################################

class PolytopicPredicate(ABC):
    """Class to define polytopic predicate containing the origin"""
    def __init__(self, polytope_0: pc.Polytope , 
                       center:     np.ndarray, 
                       is_parametric: bool=False) -> None:
        """_summary_

        Args:
            polytope_0 (pc.Polytope):  zero centered polytope 
            center (np.ndarray):       polytope center
            is_parametric (bool, optional): defines is a predicate has to be considered parametric or not. Defaults to False.

        Raises:
            RuntimeError: _description_
        """
        
        
        # when the predicate is parameteric, then the center is assumed to be the one assigned to the orginal predicate from which the predicate is derived for the decomspotion
        
        if center.ndim == 1:
            center = center.expand_dims(1) 
            

        self._polytope   = polytope_0
        if self._polytope.contains(np.zeros((self._polytope.A.shape[0],1))):
            raise ValueError("The center of the polytope must be inside the polytope")
        
        self._center     = center # center of the polygone
        self._num_hyperplanes , self._state_space_dim = np.shape(A)
        
        
        # turn the center as a column
        if np.ndim(self._center) == 0:
            self._center = self._center[:,np.newaxis]
            
        try :
            self._vertices    = [ np.expand_dims(vertex,1) for vertex in  [*pc.extreme(self._polytope)]    ] # unpacks vertices as a list of column vectors
            self._num_vertices = len(self._vertices) 
        except:
            raise RuntimeError("There was an error in the computation of the vertices for the polytope. Make sure that your polytope is closed since this is the main source of failure for the algorithm")

        self._is_parametric     = is_parametric
    
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
    

class IndependentPredicate(PolytopicPredicate):
    def __init__(self,polytope_0: pc.Polytope , 
                      center:     np.ndarray, 
                      agent_id: UniqueIdentifier,
                      is_parametric: bool=False) -> None:
        
        # initialize parent predicate
        super().__init(polytope_0 ,center,is_parametric)
        
        self._contributing_agents = [agent_id]
        
        
class CollaborativePredicate(PolytopicPredicate):
    def __init__(self,polytope_0: pc.Polytope , 
                      center:     np.ndarray, 
                      source_agent_id: UniqueIdentifier,
                      target_agent_id: UniqueIdentifier,
                      is_parametric: bool=False) -> None:
        
        # initialize parent predicate
        super().__init(polytope_0 ,center,is_parametric)
        
        self._source_agent_id = source_agent_id
        self._target_agent_id  = target_agent_id
        
        
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
        self._center = - self._center
        # change matrix A
        self._polytope = pc.Polytope(-self._polytope.A,self._polytope.b)
        
    @property
    def source_agent(self):
        return self._source_agent_id
    @property
    def target_agent(self):
        return self._target_agent_id
    @property
    def contributing_agents(self):
        return [self._source_agent_id,self._target_agent_id]
        

class StlTask:
    """STL TASK"""
    
    def __init__(self,temporal_operator:TemporalOperator, predicate:PolytopicPredicate):
        
        """
        Args:
            temporal_operator (TemporalOperator): temporal operator of the task (includes time interval)
            predicate (PolytopicPredicate): predicate of the task
        """
        
        # if a predicate function is not assigned, it is considered that the predicate is parametric
        
        self._predicate              :PolytopicPredicate  = predicate
        self._temporal_operator      :TemporalOperator    = temporal_operator
        self._parent_task            :"StlTask"           = None  # instance of the parent class
        
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
    def parent_task_id(self):
        if self._parent_task_id is None :
            raise ValueError("The task does not have a parent task specified")
        return id(self._parent_task)
    
    @property
    def  parent_task(self):
        if self._parent_task is None :
            raise ValueError("The task does not have a parent task specified")
        return self._parent_task
    
    def flip(self) :
        """Flips the direction of the predicate"""
        if not isinstance(self._predicate,CollaborativePredicate) :
            raise ValueError("The task is not a collaborative task. Individual tasks cannot be flipped")
        self._predicate.flip()
        
    def set_parent_task(self,parent_task:"StlTask")-> None:
        
        if not self._predicate.is_parametric :
            raise Warning("The task is not parametric. The parent task setting will be ignored")
        
        self._parent_task = parent_task
       
    

def create_parametric_collaborative_task_from(task : StlTask, source_agent_id:UniqueIdentifier, target_agent_id : UniqueIdentifier) -> StlTask :
    """Creates a parametric collaborative task from a given collaborative task, with anew source and target agents"""
    
    if isinstance(task.predicate,IndependentPredicate) :
        raise ValueError("The task is not a collaborative task. Individual tasks are not supported")
    
    polytope          = task.predicate.polytope
    center            = task.predicate.center
    temporal_operator = task.temporal_operator
    
    predicate =  CollaborativePredicate(polytope_0      = polytope , 
                                  center          = center, 
                                  source_agent_id = source_agent_id,
                                  target_agent_id = target_agent_id,
                                  is_parametric   = True)
    
    child_task:StlTask = StlTask(temporal_operator = temporal_operator, predicate = predicate)
    child_task.set_parent_task(parent_task = task)
    
    return child_task



def get_M_and_Z_matrices_from_inclusion(P_including:StlTask|PolytopicPredicate, P_included:StlTask|PolytopicPredicate) -> tuple[np.ndarray,np.ndarray]:
    
    if isinstance(P_including,StlTask) :
        P_including : PolytopicPredicate = P_including.predicate
    if isinstance(P_included,StlTask) :
        P_included : PolytopicPredicate = P_included.predicate
    
    
    vertices        = P_included.vertices
    num_vertices    = P_included.num_vertices
    state_space_dim = P_included.state_space_dim
    
    M = []
    for vertex in vertices:
        G_k = np.hstack((np.eye(state_space_dim),vertex))
        M.append(P_including.polytope.A@ G_k)
    
    M     = np.vstack(M)
    A_bar = np.hstack((P_including.polytope.A, P_including.polytope.b))
    Z     = np.outer(np.ones((num_vertices,1)),A_bar)    
    
    return M,Z



def communication_consistency_matrices_for(task:StlTask) -> list[np.ndarray]:
    
    vertices = task.predicate.vertices
    
    N = []
    for vertex in vertices:
        Nk = vertex.T@vertex
        N += [Nk]
    
    return  N


def random_2D_polytope(numberHyperplanes : int, distanceFromCenter: float,center:np.ndarray) -> PolytopicPredicate:
    
    numberHyperplanes = int(numberHyperplanes) # convert to int
    
    if distanceFromCenter<=0 :
        raise ValueError("distanceFromCenter must be a positive number")
    if numberHyperplanes<=2 :
        raise ValueError("number of hiperplames need to be higher than 2 in two dimensions")
    
    step = 360/numberHyperplanes
    A = np.zeros((numberHyperplanes,2))
    z = np.ones((numberHyperplanes,1))*distanceFromCenter
    
    # S =  -1+ 0.5*np.diag(np.random.random(2))
    S = np.eye(2)
    theta = 2*np.pi*np.random.rand()
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
        A[jj,:] = np.squeeze(R@S@np.array([[np.sin(angle)],[np.cos(angle)]]))
    
    
    return PolytopicPredicate(A = A, b = z,center=center)
    


def regular_2D_polytope(numberHyperplanes : int, distanceFromCenter: float,center:np.ndarray) -> PolytopicPredicate :
    
    numberHyperplanes = int(numberHyperplanes) # convert to int
    
    if distanceFromCenter<=0 :
        raise ValueError("distanceFromCenter must be a positive number")
    if numberHyperplanes<=2 :
        raise ValueError("number of hiperplames need to be higher than 2 in two dimensions")
    
    step = 360/numberHyperplanes
    A = np.zeros((numberHyperplanes,2))
    z = np.ones((numberHyperplanes,1))*distanceFromCenter
    theta = 2*np.pi*np.random.rand()
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    for jj,angle in enumerate(np.deg2rad(np.arange(0,360,step))) :
       A[jj,:] = np.squeeze(R@np.array([[np.sin(angle)],[np.cos(angle)]]))
    
    return PolytopicPredicate(A = A, b = z,center=center)



def normal_form(A : np.ndarray,b : np.ndarray) :
    
    
    for jj in range(len(b)) :
        normAi = np.sqrt(np.sum(A[jj,:]**2))
        b[jj]   /= normAi
        A[jj,:] /= normAi

        
        if b[jj] <0 :
            b[jj]   = - b[jj]
            A[jj,:] = - A[jj,:]
    
    return A,b


def rotation_matrix_2D(angle:float):
    """rotation angle in radians"""
    R = np.array([ np.cos(angle),np.sin(angle),
                  -np.sin(angle),np.cos(angle)])
    
             
if __name__ =="__main__" :
    pass
    
    
