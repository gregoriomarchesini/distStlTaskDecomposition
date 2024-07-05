import networkx as nx
from   typing import TypeVar
from   stlddec.stl_task import StlTask, CollaborativePredicate, IndependentPredicate
from copy import deepcopy
W = TypeVar("W")


MANAGER = "manager"
AGENT   = "agent"

def edge_to_int(t:tuple[int,int]) -> int :
    """Converts a tuple to an integer"""
    t= sorted(t)
    return int("".join(str(i) for i in t))


class EdgeTaskManager() :
    
    def __init__(self, edge_i :int,edge_j:int,weight:float=1) -> None:
     
      if weight<=0 : # only accept positive weights
          raise("Edge weight must be positive")
      
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
        
        if not isinstance(task,StlTask) :
            raise ValueError("The task must be a StlTask object")
        
        if isinstance(task.predicate,CollaborativePredicate) :
            source = task.predicate.source_agent
            target = task.predicate.target_agent
        elif isinstance(task.predicate,IndependentPredicate) :
            source = task.predicate.agent_id
            target = task.predicate.agent_id
        else :
            raise ValueError("The predicate must be either a CollaborativePredicate or an IndependentPredicate")
            
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
        


def create_task_graph_from_edges(edges : list[tuple[int,int]]) :
    """ Creates a communication graph from a list of edges. The edges are assumed to be undirected and all communicating"""
    
    G = nx.Graph()
    try :
        for edge in edges :
            G.add_edge(edge[0],edge[1])
            G[edge[0]][edge[1]][MANAGER] = EdgeTaskManager(edge_i =edge[0],edge_j = edge[1])
    except Exception as e:
        raise ValueError("The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The following exception was raised : \n {e}")
    
    return G

def create_communication_graph_from_edges(edges : list[tuple[int,int]],add_task_manager:bool = False) :
    """ Creates a communication graph from a list of edges. The edges are assumed to be undirected and all communicating"""
    
    G = nx.Graph()
    try :
        for edge in edges :
            G.add_edge(edge[0],edge[1])
            G[edge[0]][edge[1]][MANAGER] = EdgeTaskManager(edge_i =edge[0],edge_j = edge[1])
    except Exception as e:
        raise ValueError("The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The following exception was raised : \n {e}")
    
    return G


def normalize_graphs(comm_graph:nx.Graph, task_graph:nx.Graph) :
    """ Makes sure that both graphs have the number of edges"""
    nodes = set(comm_graph.nodes).union(set(task_graph.nodes))
    for node in nodes :
        if node not in comm_graph.nodes :
            comm_graph.add_node(node)
        if node not in task_graph.nodes :
            task_graph.add_node(node)


def create_task_graph_by_breaking_the_edges(communication_graph:nx.Graph,broken_edges:list[tuple[int,int]]) :
    """ Breaks the communication between the given edges. The edges are assumed to be undirected. The graph is not copied by the functions so
        G = break_communication_edge(G,edges) will modify the graph G as well as simply calling break_communication_edge(G,edges) will.
    """
    
    task_graph = nx.Graph()
    task_graph.add_nodes_from(communication_graph.nodes)
    
    
    
    try : 
        for edge in communication_graph.edges :
            if not (edge in broken_edges) :
                task_graph.add_edge(edge[0],edge[1])
                task_graph[edge[0]][edge[1]][MANAGER] = EdgeTaskManager(edge_i =edge[0],edge_j = edge[1])

    except Exception as e:
        raise ValueError(f"The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The exception rised was the following : \n {e}")

    return task_graph


def extract_computing_graph_from_communication_graph(comm_graph:nx.Graph) :
    
    computing_graph = nx.Graph()
    if not nx.is_tree(comm_graph) :
        raise ValueError("The communication graph must be acyclic to obtain a valid computation graph")
    
    for edge in comm_graph.edges :
        computing_graph.add_node(edge_to_int(edge))
    
    for edge in comm_graph.edges :    
        # Get all edges connected to node1 and node2
        edges_node1 = set(comm_graph.edges(edge[0]))
        edges_node2 = set(comm_graph.edges(edge[1]))
    
        # Combine the edges and remove the original edge
        adjacent_edges = list((edges_node1 | edges_node2) - {edge, (edge[1],edge[0])})
        computing_edges = [ (edge_to_int(edge), edge_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    return computing_graph
    
    
def clean_task_graph(task_graph:nx.Graph) :
    """ Removes all the tasks from the graph"""
    
    for edge in task_graph.edges :
        if not task_graph[edge[0]][edge[1]][MANAGER].has_specifications :
            task_graph.remove_edge(edge[0],edge[1])