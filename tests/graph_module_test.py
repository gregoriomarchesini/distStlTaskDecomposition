import matplotlib.pyplot as plt
from   stlddec.stl_task import regular_2D_polytope, TimeInterval, AlwaysOperator,StlTask
import polytope as pc
import networkx as nx
from   stlddec.stl_task import StlTask, CollaborativePredicate
import stlddec.graphs as gmod
import stlddec.decomposition as dmod

# create some edges
communicating_edges = [(1,2),(2,4),(1,3),(3,10),(10,5),(1,8),(8,9),(1,6),(6,7)]
task_edges          = [(1,9),(1,7),(1,4),(1,5)]

polytope =  regular_2D_polytope(5,1)

# complete_graph = gmod.create_communication_graph_from_edges(communicating_edges+broken_edges)

comm_graph = gmod.create_communication_graph_from_edges(communicating_edges)
task_graph = gmod.create_task_graph_from_edges(task_edges )
comm_graph, task_graph = gmod.normalize_graphs(comm_graph,task_graph) # this can be used to get the same nodes in both graph without specifying all of them.


# add some random tasks 
task_edges = [(1,5),(1,7),(1,4),(1,9)]

for edge in task_edges :
    P    =  CollaborativePredicate(polytope,edge[0],edge[1])
    task = StlTask(AlwaysOperator(TimeInterval(0,10)),P)
    task_graph[edge[0]][edge[1]][gmod.MANAGER].add_tasks(task)
    

G_computing = dmod.extract_computing_graph_from_communication_graph(communication_graph = comm_graph)



fig,axs = plt.subplots(1,3,figsize=(15,5)) 

pos = nx.drawing.layout.spring_layout(nx.Graph( task_edges + communicating_edges))
pos_comm = {i:p for i,p in pos.items() if i in comm_graph.nodes}
pos_task = {i:p for i,p in pos.items() if i in task_graph.nodes}

print(comm_graph.nodes)
print(G_computing.nodes)
pos_comp = {gmod.edge_to_int((i,j)): (pos_comm[i] + pos_comm[j])/2 for i,j in comm_graph.edges}
print(pos_comp)

nx.draw(comm_graph,with_labels=True,ax= axs[0],pos=pos_comm)
nx.draw(task_graph,with_labels=True,ax= axs[1],pos=pos_task)
nx.draw(G_computing,with_labels=True,ax= axs[2],pos=pos_comp)

axs[0].set_title("Communication Graph")
axs[1].set_title("Task Graph")
axs[2].set_title("Computing Graph")


plt.show()

