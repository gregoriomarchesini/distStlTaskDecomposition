import polytope as pt
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import itertools as iter
import networkx as nx
from src.decomposition_module import edgeMapping

class treeExpansionAgent() :
    def __init__(self,id:int,x:float,y:float) -> None:
        self.hasExpansionToken = False
        self.Id = id
        self.x = x
        self.y = y


numAgents = 100

completeGraph = nx.complete_graph(numAgents )
tree          = nx.Graph()


box = [[-10,10],[-10,10]]
boxwidth = box[0][1] - box[0][0]
boxheight = box[1][1] - box[1][0]

Xrand  = np.zeros((numAgents,1))
Yrand  = np.zeros((numAgents,1))
agents : list[treeExpansionAgent]= []
pos = {}

for jj in range(numAgents ) :
    xrand = box[0][0] + (boxwidth)*np.random.rand()
    yrand = box[1][0] + boxheight *np.random.rand() 
    Xrand[jj] = xrand
    Yrand[jj] = yrand
    pos[jj] = np.array([xrand,yrand])
    agent = treeExpansionAgent(id=jj,x = xrand,y = yrand)
    agents.append(agent)
    


edges  = edgeMapping()
maxDistance = 7
graph = nx.Graph()
maxIterations = 200
counter = 0

neigboursMap : dict[treeExpansionAgent,list[treeExpansionAgent]] = {}

for agenti in agents :
    neigbours = [agentj for agentj in agents if  ((((agenti.x-agentj.x)**2 + (agenti.y-agentj.y)**2) <= maxDistance**2) and agenti.Id != agentj.Id)]
    neigboursMap[agenti] = neigbours
    
#randomly select the first agent to initiate the tree expansion
agents[10].hasExpansionToken = True

for kk in range(200) :
    expandinAgents = [agent for agent in agents if agent.hasExpansionToken]
    
    for agent in expandinAgents :
        randomNeigbourAgent = np.random.choice(neigboursMap[agent])
        if not randomNeigbourAgent.hasExpansionToken :
            tree.add_edge(u_of_edge=agent.Id,v_of_edge=randomNeigbourAgent.Id)
            randomNeigbourAgent.hasExpansionToken = True # turn on the expansion agent

fig,ax = plt.subplots()
nx.draw_networkx(tree,pos=pos)




fig,ax = plt.subplots()
ax.scatter(Xrand,Yrand,c="r",marker="o",s=20)
plt.show()



        
        
                
                
            
        
        
          
