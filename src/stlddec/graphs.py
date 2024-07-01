

from stl_task import StlTask, CollaborativePredicate, IndependentPredicate
import networkx as nx
from typing import Generic, TypeVar
W = TypeVar("W")





def tuple_to_int(t:tuple) -> int :
    """Converts a tuple to an integer"""
    tuple = tuple(sorted(t))
    return int("".join(str(i) for i in t))



class UndirectedEdgeMapping(dict, Generic[W]):
    """Helper dictionary class: takes undirected edges as input keys only"""
    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            raise TypeError("Keys must tuples of type (int,int)")
        if not len(key) == 2:
            raise TypeError("Keys must tuples of type (int,int)")
        return tuple(sorted(key))
    
    def __setitem__(self, key, value):
        normalized_key = self._normalize_key(key)
        if not isinstance(value, self.__orig_class__.__args__[0]):
            raise TypeError(f"This dictionary only allows values of type {self.__orig_class__.__args__[0]}")
        dict.__setitem__(self,normalized_key, value)

    def __getitem__(self, key):
        normalized_key = self._normalize_key(key)
        return dict.__getitem__(self,normalized_key)

    def __delitem__(self, key):
        normalized_key = self._normalize_key(key)
        dict.__delitem__(self,normalized_key)

    def __contains__(self, key):
        normalized_key = self._normalize_key(key)
        return dict.__contains__(self,normalized_key)
    

    
class GraphEdge() :
    
    def __init__(self, edge_i :int,edge_j:int, is_communicating :int= 0,weight:float=1) -> None:
     
      if weight<=0 : # only accept positive weights
          raise("Edge weight must be positive")
      
      self._is_communicating            = is_communicating
      self._is_involved_in_optimization = 0
      
      if not(self._is_communicating) :
          self._weight = float("inf")
      else :
          self._weight = weight
          
      self._tasks_list      = []  
      
      if (not isinstance(edge_i,int)) or (not isinstance(edge_j,int)) :
          raise ValueError("Target source pairs must be integers")
      else :
          self._edge = (edge_i,edge_j)
      
    @property
    def tasks_list(self) :
        return self._tasks_list
    
    @property
    def is_communicating(self) :
      return self._is_communicating 
   
    @property
    def edge(self) :
        return self._edge
  
    @property
    def is_involved_in_optimization(self) :
      return self._is_involved_in_optimization
  
    @property
    def weight(self):
        return self._weight
    
    @property
    def has_specifications(self):
        return bool(len(self._tasks_list)) 

    @weight.setter
    def weight(self,new_weight:float)-> None :
        if not isinstance(new_weight,float) :
            raise TypeError("Weight must be a float")
        elif new_weight<0 :
            raise ValueError("Weight must be positive")
        else :
            self._weight = new_weight
    
    
    def is_task_consistent_with_this_edge(self,task:StlTask) -> bool :
        
        if isinstance(task.predicate,CollaborativePredicate) :
            source = task.predicate.source_agent
            target = task.predicate.target_agent
        elif isinstance(task.predicate,IndependentPredicate) :
            source = task.predicate.agent_id
            target = task.predicate.agent_id
            
        return ( (source,target) == self._edge ) or ( (target,source) == self._edge )
    
    
   
    def _add_single_task(self,input_task : StlTask) -> None :
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not (isinstance(input_task,StlTask)) :
            raise Exception("Please enter a valid STL task object or a list of StlTask objects")
        
        
        if self.is_task_consistent_with_this_edge(task = input_task) :
            self._tasks_list.append(input_task)
        else :
            raise ValueError(f"The task is not consistent with the edge. Task is defined over the edge {input_task.predicate.source_agent,input_task.predicate.target_agent} while the edge is defined over {self._edge}")
            
            
    def add_tasks(self,tasks : StlTask|list[StlTask]):
        
        if isinstance(tasks,list) : # list of tasks
            for  task in tasks :
                self._add_single_task(task)
        else : # single task case
            self._add_single_task(tasks)
    
 
    def flag_optimization_involvement(self) -> None :
        self._is_involved_in_optimization = True
        
        
        
def compute_weights(source,target,attributesDict) :
    """takes the edge object from the attributes and returns the wight stored in there"""
    return attributesDict["edgeObj"].weight    


def create_communication_graph_from_edges(edge_map :  UndirectedEdgeMapping[GraphEdge]) :
    """ Builds a undirected graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    comm_graph = nx.Graph()
    
    for edge,edge_obj in edge_map.items() :
        if edge_obj.is_communicating : 
            comm_graph.add_edge(edge[0],edge[1])
    return comm_graph


def create_task_graph_from_edges(edge_map :  UndirectedEdgeMapping[GraphEdge]):
    """ Builds a undirected graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    task_graph = nx.Graph()
    
    for edge,edge_obj in edge_map.items() :
        if edge_obj.has_specifications: 
            task_graph.add_edge(edge[0],edge[1])
    return task_graph
    
    
def create_computing_graph_from_communication_graph(comm_graph:nx.Graph) :
    
    computing_graph = nx.Graph()
    if not nx.is_tree(comm_graph) :
        raise ValueError("The communication graph must be acyclic to obtain a valid computation graph")
    
    for edge in comm_graph.edges :
        
        computing_graph.add_node(tuple_to_int(edge))
    
    for edge in comm_graph.edges :    
        # Get all edges connected to node1 and node2
        edges_node1 = set(comm_graph.edges.edges(edge[0]))
        edges_node2 = set(comm_graph.edges.edges(edge[1]))
    
        # Combine the edges and remove the original edge
        adjacent_edges = list((edges_node1 | edges_node2) - {edge})
        computing_edges = [ (tuple_to_int(edge), tuple_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    return computing_graph
    
if __name__ == "__main__" :
    
    uem = UndirectedEdgeMapping[str]()

    try:
        uem[(1, 2)] = 2  # This should raise a TypeError
    except TypeError as e:
        print(e)