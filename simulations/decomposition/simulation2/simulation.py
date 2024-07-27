
import numpy as np
import matplotlib.pyplot as plt
import stl.stl as pmod
import stlddec.graphs as gmod
import stlddec.decomposition as dmod
import polytope as pc
import networkx as nx
import networkx as nx


#! Fix example and do more complicated examples:
#! 1. Add more agents
#! 2. check what happens to the contraint when you have a lot of overloading of agents
#! 3. check if the cost 1/scale is better than the linear cost (it should be)
#! 4. check if scaling of the variables help even if I doubt since the center variables do not have crazy high values.





# List all the edges in the network with communication
task_edges = [(1,2),(2,3),(3,1)]
comm_edges = [(1,2),(3,1)]     

task_graph = gmod.create_task_graph_from_edges(task_edges)
comm_graph = gmod.create_communication_graph_from_edges( comm_edges)

task_edges = [(1,2),(2,3),(3,1)]
comm_edges = [(1,2),(3,1)]     

task_graph = gmod.create_task_graph_from_edges(task_edges)
comm_graph = gmod.create_communication_graph_from_edges( comm_edges)


# -------------- Setting the tasks -----------------------
# EDGE 12
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(5,4)
    
# Create a predicate!
predicate   = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([5,-5]),
                                           source_agent_id=1,
                                           target_agent_id=2)
# Set a temporal operator.
t_operator  = pmod.G(pmod.TimeInterval(0,10))
print(t_operator)
print(t_operator)
    
# Create a task.
task        = pmod.StlTask(temporal_operator=t_operator,predicate=predicate)
    
# Add the task to the edge.
task_graph[1][2][gmod.MANAGER].add_tasks(task)
task_graph[1][2][gmod.MANAGER].add_tasks(task)

# EDGE 32
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(5,4)
    
# Create a predicate!
predicate   = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([0,0]),
                                           source_agent_id=3,
                                           target_agent_id=2)
# Set a temporal operator.
t_operator  = pmod.G(pmod.TimeInterval(0,10))
    
# Create a task.
task        = pmod.StlTask(temporal_operator=t_operator,predicate=predicate)
    
# Add the task to the edge.
task_graph[3][2][gmod.MANAGER].add_tasks(task)
task_graph[3][2][gmod.MANAGER].add_tasks(task)
    

# -------------- Run The Decomposition -----------------------
new_task_graph, edge_computing_graph = dmod.run_task_decomposition(communication_graph = comm_graph, task_graph = task_graph,number_of_optimization_iterations =1000, communication_radius=20,logger_level="ERROR")


fig,axs = plt.subplots(1,4,figsize=(15,5)) 

pos = {1:np.array([0,0]),2:np.array([1,1]),3:np.array([2,0])}
pos_comp     = {gmod.edge_to_int((i,j)): (pos[i] + pos[j])/2 for i,j in comm_graph.edges}


nx.draw(comm_graph,with_labels=True    ,ax= axs[0],pos=pos)
nx.draw(task_graph,with_labels=True    ,ax= axs[1],pos=pos)
nx.draw(new_task_graph,with_labels=True,ax= axs[2],pos=pos)
nx.draw(edge_computing_graph,with_labels=True,ax= axs[3],pos=pos_comp)

axs[0].set_title("Communication Graph")
axs[1].set_title("Task Graph")
axs[2].set_title("New Task Graph")
axs[3].set_title("Computing Graph")




plt.show()

