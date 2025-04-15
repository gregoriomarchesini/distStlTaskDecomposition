import networkx as nx
from   typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from  .stl import StlTask, CollaborativePredicate, IndependentPredicate



MANAGER = "manager"
AGENT   = "agent"

def edge_to_int(t:tuple[int,int]) -> int :
    """Converts a tuple to an integer"""
    t= sorted(t,reverse=True) # to avoid node "02" to becomes 2 we reverse order
    return int("".join(str(i) for i in t))



class TaskGraph(nx.Graph) :
    def __init__(self,incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
    
    def add_edge(self,u_of_edge, v_of_edge) :
        """ Adds an edge to the graph if it is not already present"""
        if not (u_of_edge,v_of_edge) in self.edges :
            super().add_edge(u_of_edge, v_of_edge)
            self[u_of_edge][v_of_edge][MANAGER] = EdgeTaskManager(edge_i = u_of_edge,edge_j = v_of_edge)
    
    def add_edges_from(self, ebunch_to_add) :
        """ Adds multiple edges to the graph."""
        for edge in ebunch_to_add :
            self.add_edge(edge[0],edge[1])
    
    def _attach_single(self,task: StlTask) :
        """ Attaches a task to the edge of the graph"""
        if not isinstance(task,StlTask) :
            raise ValueError("The task must be a StlTask object")
        
        if isinstance(task.predicate,IndependentPredicate) :
            self.add_edge(task.predicate.agent_id,task.predicate.agent_id) # it will add the edge if not present
            self.edges[task.predicate.agent_id,task.predicate.agent_id][MANAGER].add_tasks(task)
            
        elif isinstance(task.predicate,CollaborativePredicate) :
            self.add_edge(task.predicate.source_agent,task.predicate.target_agent) # it will add the edge if not present
            self.edges[task.predicate.source_agent,task.predicate.target_agent][MANAGER].add_tasks(task)
            
        else :
            raise ValueError(f"The given preidcate of type {type(task.predicate)} is not supported. Supported predicate types are {CollaborativePredicate.__name__} and {IndependentPredicate.__name__}")
        
    def attach(self,tasks:StlTask|list[StlTask]) :
        """ Attaches a task to the edge of the graph"""
        if isinstance(tasks,list) :
            for task in tasks :
                self._attach_single(task)
        else :
            self._attach_single(tasks)
            
    def task_list_for_edge(self,u_of_edge, v_of_edge) -> list[StlTask] :
        """ Returns the list of tasks for the given edge"""
        return self.edges[u_of_edge, v_of_edge][MANAGER].tasks_list
            

class CommunicationGraph(nx.Graph) :
    def __init__(self,incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
        
    
    def add_edge(self,u_of_edge, v_of_edge,**attr) :
        """ Adds an edge to the graph if it is not already present"""
        if not (u_of_edge,v_of_edge) in self.edges :
            super().add_edge(u_of_edge, v_of_edge,**attr)
            
        if not (u_of_edge == v_of_edge) :
            super().add_edge(v_of_edge,v_of_edge)
            super().add_edge(u_of_edge, u_of_edge)
    
    def add_edges_from(self, ebunch_to_add,**attr) :
        """ Adds multiple edges to the graph."""
        for edge in ebunch_to_add :
            self.add_edge(edge[0],edge[1],**attr)

class EdgeTaskManager() :
    
    def __init__(self, edge_i :int,edge_j:int,weight:float=1) -> None:
     
      if weight<=0 : # only accept positive weights
          raise("Edge weight must be positive")
      
      self._weight = weight    
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
            raise ValueError(f"The predicate must be either a {CollaborativePredicate.__name__} or an {IndependentPredicate.__name__}")
            
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
        

def create_task_graph_from_edges(edges : list[tuple[int,int]])-> TaskGraph :
    """ Creates a communication graph from a list of edges. The edges are assumed to be undirected and all communicating"""
    
    G = TaskGraph()
    try :
        for edge in edges :
            G.add_edge(edge[0],edge[1])
    except Exception as e:
        raise ValueError(f"The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The following exception was raised : \n {e}")
    
    return G

def create_communication_graph_from_edges(edges : list[tuple[int,int]],add_task_manager:bool = False) -> CommunicationGraph:
    """ Creates a communication graph from a list of edges. The edges are assumed to be undirected and all communicating"""
    
    G = CommunicationGraph()
    try :
        for edge in edges :
            G.add_edge(edge[0],edge[1])
    except Exception as e:
        raise ValueError(f"The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The following exception was raised : \n {e}")
    
    return G

def normalize_graphs(*graphs : Union[CommunicationGraph,TaskGraph]) -> tuple[CommunicationGraph,TaskGraph]:
    """ Makes sure that both graphs have the number of edges"""
    
    
    nodes = set()
    for graph in graphs :
        nodes.union(set(graph.nodes))
    
    for graph in graphs :
      graph.add_nodes_from(nodes)
    
    return graphs

def create_task_graph_by_breaking_the_edges(communication_graph:CommunicationGraph,broken_edges:list[tuple[int,int]]) -> TaskGraph:
    """ Breaks the communication between the given edges. The edges are assumed to be undirected. The graph is not copied by the functions so
        G = break_communication_edge(G,edges) will modify the graph G as well as simply calling break_communication_edge(G,edges) will.
    """
    
    task_graph = TaskGraph()
    task_graph = TaskGraph()
    task_graph.add_nodes_from(communication_graph.nodes)
    
    
    try : 
        for edge in communication_graph.edges :
            if not (edge in broken_edges) :
                task_graph.add_edge(edge[0],edge[1])
    except Exception as e:
        raise ValueError(f"The edges must be a list of tuples. EX: [(1,2), (2,3), ...]. The exception rised was the following : \n {e}")

    return task_graph



    
def clean_task_graph(task_graph:TaskGraph) -> TaskGraph:
    """ Removes edges that do not have specifications"""
    
    if not isinstance(task_graph,TaskGraph) :
        raise ValueError("The input must be a TaskGraph object")

    
def clean_task_graph(task_graph:TaskGraph) -> TaskGraph:
    """ Removes edges that do not have specifications"""
    
    if not isinstance(task_graph,TaskGraph) :
        raise ValueError("The input must be a TaskGraph object")
    for edge in task_graph.edges :
        if not task_graph[edge[0]][edge[1]][MANAGER].has_specifications :
            task_graph.remove_edge(edge[0],edge[1])
    return task_graph


def get_regular_polytopic_tree_graph(num_vertices : int, num_polygones : int, inter_ring_distance:float = 9)-> tuple[CommunicationGraph,TaskGraph, dict[int,np.ndarray]]:
    """
    
    Create a star communication graph and a concenstric ring task graph. The number of vertices defined the shape of the concentric
    polytogones and the number of polygones defines how many concentric polygones you have. For example if you call the function with
    num_vertices = 4 and num_polygones = 2 you will get this graph: 
    
    7-----------6
    |           |
    |  3-----2  |
    |  |     |  |
    |  |     |  |
    |  |  0  |  |
    |  |     |  |
    |  |     |  |
    |  4-----1  |
    |           |
    8-----------5 
    
    The task graph will contain all the edges in the rings (so it is two disconnected rings) and the communication graph will be the star graph with edges
    connecting the two rings and the center.

    Args:
        num_vertices (int) : number of vertices per each polytope ring
        num_polygones (int): number of rings around the center
        dist_from_center (float): distance from the center of the star graph to the first ring

    Returns:
        comm_graph (CommunicationGraph) 
        task_graph (TaskGraph)
        pos        (dict[int,np.ndarray]) : dictionary containing the position of each node (used for plotting the graph)
    """
    
    if inter_ring_distance < 0 :
        raise ValueError("Distance from center must be positive")
    if num_vertices < 3 :
        raise ValueError("The number of vertices must be at least 3")
    if num_polygones < 1 :
        raise ValueError("The number of polygones must be at least 1")
    
    polygones_nodes = []
    pos             = {}
    
    
    for i in range(0,num_polygones) :
        nodes = list(range(1+ num_vertices*i, (num_vertices+1) + num_vertices*i))
        
        for node,kk in zip(nodes,range(0,num_vertices)) :
            pos[node] = (i+1)*inter_ring_distance*np.array([np.cos( 2*np.pi/num_vertices*kk - np.pi/4 ),np.sin(2*np.pi/num_vertices*kk - np.pi/4 )])

        polygones_nodes .append(nodes)

    perimeter_edges_per_polytope = []
    for i in range(num_polygones) :
        for j in range(num_vertices) :
            perimeter_edges_per_polytope.append((polygones_nodes[i][j-1],polygones_nodes[i][j]))

    interpolitope_vertices = []
    for i in range(num_polygones-1) :
        for vertex in range(num_vertices) :
            interpolitope_vertices.append((polygones_nodes[i][vertex ],polygones_nodes[i+1][vertex ]))
        
    # add center coonection to first ring 
    for vertex in range(num_vertices) :
        interpolitope_vertices.append((polygones_nodes[0][vertex],0))
    
    
    star_edges = interpolitope_vertices + [ (polygones_nodes[0][vertex],0) for vertex in range(0,num_vertices)]
    rings_vertices =  perimeter_edges_per_polytope
    pos[0] = np.array([0,0])
    
    comm_graph = create_communication_graph_from_edges(star_edges)
    task_graph = create_task_graph_from_edges(rings_vertices)
    comm_graph, task_graph = normalize_graphs(comm_graph,task_graph) # this can be used to get the same nodes in both graph without specifying all of them.

    return comm_graph, task_graph, pos


def show_graphs(*graphs, titles:list[str] = []) :
    """ Plots the graphs"""

    if not len(titles) == len(graphs) :
        raise ValueError("The number of labels must be equal to the number of graphs")
    
    random_n = lambda : 0.5-np.random.rand()
    pos = nx.spring_layout(graphs[0])
    pos = {k: np.array([pos[k][0]*(1.+random_n()),pos[k][1]*(1.+random_n())]) for k in pos.keys()}
    fig,axs = plt.subplots(1,len(graphs),figsize=(15,5)) 
    
    for i,graph in enumerate(graphs) :
        nx.draw(graph,with_labels=True,ax= axs[i],pos=pos)
        axs[i].set_title(titles[i])
        
        

def visualize_tasks_in_the_graph(task_graph: TaskGraph) :
    
    num_edges = len(task_graph.edges)
    n_cols = 4
    n_rows = int(num_edges/n_cols) +1
    
    fig,axs = plt.subplots(n_rows,n_cols,figsize=(15,15))
    axs     = axs.flatten()
    counter = 0
    for edge in task_graph.edges :
        tasks = task_graph.task_list_for_edge(edge[0],edge[1])
        
        for i,task in enumerate(tasks) :
            if not(task.predicate.is_parametric) :
                center   = task.predicate.center         
                vertices = (np.hstack(task.predicate.vertices) + center).T
                
                
                p        = Polygon(vertices , facecolor = 'gray',alpha = 0.5, edgecolor='k')
                axs[counter].arrow(0.,0.,center[0].item(),center[1].item(),length_includes_head=True)
                axs[counter].add_patch(p)
                axs[counter].set_title(f"Edge {edge}")
        
        axs[counter].autoscale()
        counter +=1
    plt.tight_layout()