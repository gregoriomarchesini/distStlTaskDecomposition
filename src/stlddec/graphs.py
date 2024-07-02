

from stl_task import StlTask, CollaborativePredicate, IndependentPredicate
import networkx as nx
from typing import Generic, TypeVar
W = TypeVar("W")




def edge_to_int(t:tuple[int,int]) -> int :
    """Converts a tuple to an integer"""
    t= sorted(t)
    return int("".join(str(i) for i in t))


class EdgeTaskManager() :
    
    def __init__(self, edge_i :int,edge_j:int, is_communicating :bool= True,weight:float=1) -> None:
     
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
    
    @is_communicating.setter
    def is_communicating(self,value):
        if not isinstance(value,bool) :
            raise ValueError("The value must be a boolean for the property is_communicating")
        self._is_communicating = value
    
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
        

def create_graph_from_edges(edges : list[tuple[int,int]]) :
    """ Creates a communication graph from a list of edges. The edges are assumed to be undirected and all communicating"""
    
    G = nx.Graph()
    try :
        for edge in edges :
            G.add_edge(edge[0],edge[1])
            G[edge[0]][edge[1]]["tasks_store"] = EdgeTaskManager(edge_i =edge[0],edge_j = edge[1],is_communicating=True)
    except :
        raise ValueError("The edges must be a list of tuples. EX: [(1,2), (2,3), ...]")
    
    return G

def break_communication_edge(G:nx.Graph,edges:list[tuple[int,int]]) :
    """ Breaks the communication between the given edges. The edges are assumed to be undirected. The graph is not copied by the functions so
        G = break_communication_edge(G,edges) will modify the graph G as well as simply calling break_communication_edge(G,edges) will.
    """
    try : 
        for edge in edges :
            if G.has_edge(edge[0],edge[1]) :
                G[edge[0]][edge[1]]["tasks_store"].is_communicating = False
    except Exception as e:
        raise ValueError(f"The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The exception rised was the following : \n {e}")

    return G

def extract_task_graph(graph:nx.Graph) :
    """ Builds a undirected graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    task_graph = nx.Graph()
    for edge in graph.edges :
        if graph[edge[0]][edge[1]]["tasks_store"].has_specifications :
            task_graph.add_edge(edge[0],edge[1])
            task_graph[edge[0]][edge[1]]["tasks_store"] = graph[edge[0]][edge[1]]["tasks_store"]
    return task_graph

def extract_communication_graph(graph:nx.Graph) :
    """ Builds a directed graph to be used for the decomposition based on the edges. The given edges are inserted in both directions for an edge"""
    
    comm_graph = nx.Graph()
    for edge in graph.edges :
        if graph[edge[0]][edge[1]]["tasks_store"].is_communicating :
            print(edge)
            comm_graph.add_edge(edge[0],edge[1])
            comm_graph[edge[0]][edge[1]]["tasks_store"] = graph[edge[0]][edge[1]]["tasks_store"]
    return comm_graph
    
    
def get_computing_graph_from_communication_graph(comm_graph:nx.Graph) :
    
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
        adjacent_edges = list((edges_node1 | edges_node2) - {edge})
        computing_edges = [ (edge_to_int(edge), edge_to_int(edge_neigh)) for edge_neigh in adjacent_edges]
        
        computing_graph.add_edges_from(computing_edges)
    
    return computing_graph
    
if __name__ == "__main__" :
    
    import matplotlib.pyplot as plt
    import numpy as np
    from stl_task import AbstractPolytopicPredicate, regular_2D_polytope, TimeInterval, AlwaysOperator,StlTask
    import polytope as pc
    
    # create some edges
    communicating_edges = [(1,2),(2,4),(1,3),(3,10),(10,5),(1,8),(8,9),(1,6),(6,7)]
    broken_edges        = [(1,9),(1,7),(1,4),(1,5)]
    
    A,b  =  regular_2D_polytope(5,1)
    
    
    G = create_graph_from_edges(communicating_edges+broken_edges)
    G = break_communication_edge(G,broken_edges)
    
    # add some random tasks 
    task_edges = [(1,2),(1,3),(1,6),(1,8)]
    
    for edge in task_edges :
        P    =  CollaborativePredicate(pc.Polytope(A,b),edge[0],edge[1])
        task = StlTask(AlwaysOperator(TimeInterval(0,10)),P)
        G[edge[0]][edge[1]]["tasks_store"].add_tasks(task)
        
    
    G_comm = extract_communication_graph(G)
    G_task = extract_task_graph(G)
    G_computing = get_computing_graph_from_communication_graph(G_comm)
    
    
    
    fig,axs = plt.subplots(1,4,figsize=(15,5)) 
    
    pos = nx.drawing.layout.spring_layout(G)
    pos_comm = {i:p for i,p in pos.items() if i in G_comm.nodes}
    pos_task = {i:p for i,p in pos.items() if i in G_task.nodes}
    
    print(G_comm.nodes)
    print(G_computing.nodes)
    pos_comp = {edge_to_int((i,j)): (pos_comm[i] + pos_comm[j])/2 for i,j in G_comm.edges}
    print(pos_comp)
    
    nx.draw(G,with_labels=True,ax= axs[0],pos=pos)
    nx.draw(G_comm,with_labels=True,ax= axs[1],pos=pos_comm)
    nx.draw(G_task,with_labels=True,ax= axs[2],pos=pos_task)
    nx.draw(G_computing,with_labels=True,ax= axs[3],pos=pos_comp)
    
    axs[0].set_title("Full Graph")
    axs[1].set_title("Communication Graph")
    axs[2].set_title("Task Graph")
    axs[3].set_title("Computing Graph")
    
    plt.show()
    
    