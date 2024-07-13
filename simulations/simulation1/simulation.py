import numpy as np
import matplotlib.pyplot as plt
import stlddec.stl_task as pmod
import stlddec.graphs as gmod
import stlddec.decomposition as dmod
import networkx as nx
import random

np.random.seed(100)
random.seed(100)
def get_regular_polytopic_star_graph(vertices : int, num_polygones : int) :
    
    
    polygones_nodes = []
    pos             = {}
    dist_from_center = 9
    
    
    for i in range(0,num_polygones) :
        nodes = list(range(1+ vertices*i, (vertices+1) + vertices*i))
        
        for node,kk in zip(nodes,range(0,vertices)) :
            pos[node] = (i+1)*dist_from_center*np.array([np.cos( 2*np.pi/vertices*kk - np.pi/4 ),np.sin(2*np.pi/vertices*kk - np.pi/4 )])
     
        
        

        polygones_nodes .append(nodes)

    perimeter_edges_per_polytope = []
    for i in range(num_polygones) :
        for j in range(vertices) :
            perimeter_edges_per_polytope.append((polygones_nodes[i][j-1],polygones_nodes[i][j]))

    interpolitope_vertices = []
    for i in range(num_polygones-1) :
        for vertex in range(vertices) :
            interpolitope_vertices.append((polygones_nodes[i][vertex ],polygones_nodes[i+1][vertex ]))
        
    # add center coonection to first ring 
    for vertex in range(vertices) :
        interpolitope_vertices.append((polygones_nodes[0][vertex],0))
    
    
    star_edges = interpolitope_vertices + [ (polygones_nodes[0][vertex],0) for vertex in range(0,vertices)]
    rings_vertices =  perimeter_edges_per_polytope
    pos[0] = np.array([0,0])
    
    comm_graph = gmod.create_communication_graph_from_edges(star_edges)
    task_graph = gmod.create_task_graph_from_edges(rings_vertices)
    comm_graph, task_graph = gmod.normalize_graphs(comm_graph,task_graph) # this can be used to get the same nodes in both graph without specifying all of them.

    return comm_graph, task_graph, pos


comm_graph, task_graph,regular_positions = get_regular_polytopic_star_graph(vertices = 4,num_polygones=3)
print(regular_positions)

# ------ adding some tasks at random -------- 
tasking_percentage = 0.5
edges_to_be_tasked = random.sample( list(task_graph.edges), int(len(task_graph.edges)*tasking_percentage) )
print(edges_to_be_tasked)

number_of_hyperplanes = 5
distance_from_center  = 4

for edge in edges_to_be_tasked :
    
    
    center_regular = regular_positions[edge[0]] - regular_positions[edge[1]]
    print(center_regular)
    predicate   = pmod.CollaborativePredicate(polytope_0 = pmod.regular_2D_polytope(number_of_hyperplanes,distance_from_center),
                                              source_agent_id   = edge[1],
                                              target_agent_id   = edge[0], 
                                              center            = center_regular)

    task = pmod.StlTask(temporal_operator = pmod.AlwaysOperator(pmod.TimeInterval(0,10)),
                        predicate         = predicate)
    
    task_graph[edge[0]][edge[1]][gmod.MANAGER].add_tasks(task)

task_graph = gmod.clean_task_graph(task_graph)


new_task_graph, edge_computing_graph = dmod.run_task_decomposition(communication_graph = comm_graph, task_graph = task_graph,number_of_optimization_iterations =1000, communication_radius=20,logger_level="ERROR")


fig,axs = plt.subplots(1,4,figsize=(15,5)) 

pos = nx.spring_layout(comm_graph)
pos_comp     = {gmod.edge_to_int((i,j)): (regular_positions[i] + regular_positions[j])/2 for i,j in comm_graph.edges}


nx.draw(comm_graph,with_labels=True    ,ax= axs[0],pos=regular_positions)
nx.draw(task_graph,with_labels=True    ,ax= axs[1],pos=regular_positions)
nx.draw(new_task_graph,with_labels=True,ax= axs[2],pos=regular_positions)
nx.draw(edge_computing_graph,with_labels=True,ax= axs[3],pos=pos_comp)

axs[0].set_title("Communication Graph")
axs[1].set_title("Task Graph")
axs[2].set_title("New Task Graph")
axs[3].set_title("Computing Graph")
plt.show()