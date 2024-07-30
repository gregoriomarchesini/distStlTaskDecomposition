from   enum import Enum
import multiprocessing as mp
from   networkx import Graph, is_tree
from   networkx import diameter as net_diameter
import networkx as nx
import numpy as np
import sys,os



    
class LeadershipToken(Enum) :
    LEADER    = 1
    FOLLOWER  = 2
    UNDEFINED = 0


Ti = dict[int,LeadershipToken] # token set fo a single agent


def print_tokens(tokens:dict[int,Ti]) :
    for unique_identifier,Ti in tokens.items() :
        print(f"Agent {unique_identifier} has tokens:")
        for j,token in Ti.items() :
           print(f"tau_{unique_identifier,j} = {token}")
        
def update_tokens(unique_identifier : int,tokens_dictionary : dict[int,Ti]) :
    
    Ti = tokens_dictionary[unique_identifier]
    
    for j in Ti.keys() :
        if len(Ti.keys()) == 1  : #edge leader (leaf node)
            Ti[j] = LeadershipToken.LEADER # set agent as edge leader
            
        else :   
            Tj = tokens_dictionary[j]
            if Tj[unique_identifier] == LeadershipToken.LEADER :
                Ti[j] = LeadershipToken.FOLLOWER # set agent as edge follower
    
    count = 0
    for j,token in Ti.items():
        if token == LeadershipToken.UNDEFINED :
            count+= 1
            index = j
    if count == 1 : # there is only one undecided edge
        Ti[index] = LeadershipToken.LEADER # set agent as edge leader


def any_undefined(Ti:dict[int,LeadershipToken]) -> bool :
    for token in Ti.values():
        if token == LeadershipToken.UNDEFINED :
            return True
    return False


def token_passing_algorithm(task_graph:Graph):
    """
    implements token passing algorithm. Given graph should be acyclic or the expected output will not tbe correct
    """
    # 0 undefined
    # 1 leader 
    # 2 follower
    tokens_dictionary = {}
    # parallelized version (the overhead from to little agents will not be worth it)
    
    agents = sorted(list(task_graph.nodes()), reverse=True)
    
    for agent in agents :
        Ti = {unique_identifier:LeadershipToken.UNDEFINED for unique_identifier in task_graph.neighbors(agent) if unique_identifier!=agent}
        tokens_dictionary[agent] = Ti
    
    diameter = net_diameter(task_graph)
    
    for round in range(int(np.ceil(diameter/2))+1) :
        for agent in agents :
            update_tokens(agent,tokens_dictionary)
            
    
    for Ti in tokens_dictionary.values() :
        if any_undefined(Ti) :
            print_tokens(tokens_dictionary)
            raise RuntimeError("Error: token passing algorithm did not converge")
            
    return tokens_dictionary


class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        
