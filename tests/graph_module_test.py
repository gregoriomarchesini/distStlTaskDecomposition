import matplotlib.pyplot as plt
from   stlddec.stl_task import regular_2D_polytope, TimeInterval, AlwaysOperator,StlTask
import polytope as pc
import networkx as nx
from   stlddec.stl_task import StlTask, CollaborativePredicate
import stlddec.graphs as gmod

# create some edges
communicating_edges = [(1,2),(2,4),(1,3),(3,10),(10,5),(1,8),(8,9),(1,6),(6,7)]
broken_edges        = [(1,9),(1,7),(1,4),(1,5)]

A,b  =  regular_2D_polytope(5,1)


G = gmod.create_graph_from_edges(communicating_edges+broken_edges)
G = gmod.break_communication_edge(G,broken_edges)

# add some random tasks 
task_edges = [(1,5),(1,7),(1,4),(1,9)]

for edge in task_edges :
    P    =  CollaborativePredicate(pc.Polytope(A,b),edge[0],edge[1])
    task = StlTask(AlwaysOperator(TimeInterval(0,10)),P)
    G[edge[0]][edge[1]][gmod.MANAGER].add_tasks(task)
    

G_comm      = gmod.__extract_communication_graph(G)
G_task      = gmod.__extract_task_graph(G)
G_computing = gmod.__extract_computing_graph_from_communication_graph(G_comm)



fig,axs = plt.subplots(1,4,figsize=(15,5)) 

pos = nx.drawing.layout.spring_layout(G)
pos_comm = {i:p for i,p in pos.items() if i in G_comm.nodes}
pos_task = {i:p for i,p in pos.items() if i in G_task.nodes}

print(G_comm.nodes)
print(G_computing.nodes)
pos_comp = {gmod.edge_to_int((i,j)): (pos_comm[i] + pos_comm[j])/2 for i,j in G_comm.edges}
print(pos_comp)

nx.draw(G,with_labels=True,ax= axs[0],pos=pos)
nx.draw(G_comm,with_labels=True,ax= axs[1],pos=pos_comm)
nx.draw(G_task,with_labels=True,ax= axs[2],pos=pos_task)
nx.draw(G_computing,with_labels=True,ax= axs[3],pos=pos_comp)

axs[0].set_title("Full Graph")
axs[1].set_title("Communication Graph")
axs[2].set_title("Task Graph")
axs[3].set_title("Computing Graph")

print(G.has_edge(2,1))
from itertools import combinations
agents_combinations = list(combinations(G_computing.nodes,2))
print(agents_combinations)

plt.show()

