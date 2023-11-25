from networkx import Graph,complete_graph,wheel_graph, draw_networkx,diameter, neighbors, sudoku_graph
import matplotlib.pyplot as plt
import polytope


class SearchTemplate():
    def __init__(self,target:int,source:int) -> None:
        self._target = target
        self._source = source
        self._path   = []
        self._isNeigbour = False
        
    @property
    def pathLength(self):
        return len(self._path)
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self,path) :
        self._path = path
    @property
    def target(self):
        return self._target
    
    @property
    def isNeigbour(self):
        return self._isNeigbour
    
    @isNeigbour.setter
    def isNeigbour(self,bit):
        self._isNeigbour = bit

# create a graph 

G = sudoku_graph(2)
rho = diameter(G)
draw_networkx(G)
print(G.nodes)
target = 6
source = 5


agents = {index:SearchTemplate(target=target,source = target) for index in G.nodes}

for jj in range(2*rho) :
    for agentIndex,searchTemplte in agents.items() :
        
        if target in G.neighbors(agentIndex) and (not searchTemplte.isNeigbour) :
            searchTemplte.path = [agentIndex,target]
            searchTemplte.isNeigbour = True
        
        elif not searchTemplte.isNeigbour :
            for neigbourIndex in G.neighbors(agentIndex) :
                if agents[neigbourIndex].pathLength!=0 : #one of your neigbours found a path
                    if ((agents[neigbourIndex].pathLength +1)< searchTemplte.pathLength) or (searchTemplte.pathLength==0) : #if I did not find a patrh previously or someone has a shorter path
                        searchTemplte.path = [agentIndex] + agents[neigbourIndex].path # add myself to the path
                    
for agent,searchTemplate in agents.items() :
    print(agent)
    print("path : ",searchTemplate.path)          
    
    
plt.show()