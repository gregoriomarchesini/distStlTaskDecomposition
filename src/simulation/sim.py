"""
MIT License

Copyright (c) [2024] [Gregorio Marchesini]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import matplotlib.pyplot as plt
import numpy as np
from   time import perf_counter
from   tqdm import tqdm
from   traceback import format_exc
from   dataclasses import dataclass
import networkx as nx

from stlddec.graphs     import CommunicationGraph,TaskGraph
from stlcont.controller import STLController, get_logger
from stlcont.utils      import token_passing_algorithm, LeadershipToken
from stl.dynamics       import DynamicalModel,SingleIntegrator2D


plt.rcParams["figure.figsize"] = (3.425, 2.325)

logger = get_logger(__name__)

@dataclass
class MultiAgentSystem:
    task_graph          : TaskGraph
    communication_graph : CommunicationGraph
    agents_states       : dict[int,np.ndarray]
    agents_models       : dict[int,DynamicalModel]
    current_time        : float = 0
    
    def __post_init__(self):
        
        if set(self.agents_states.keys()) != set(self.agents_models.keys()):
            raise ValueError("The keys of the agents_states and agents_models dictionaries must be the same")
        if set(self.agents_states.keys()) != set(self.communication_graph.nodes):
            raise ValueError("The keys of the agents_states and communication_graph dictionaries must be the same")



def simulate_agents(multi_agent_system : MultiAgentSystem, final_time:int) -> None:
    
    
    state_history = {unique_identifier : [] for unique_identifier in multi_agent_system.agents_states.keys()}
    time_step = DynamicalModel._time_step
    number_of_steps = int(final_time/time_step)
    
    # Create one controller for each agent.
    controllers : dict[int,STLController]= dict()
    
    # Set up the controller for each agent.
    for agent_id,model in multi_agent_system.agents_models.items():
        
        controllers[agent_id] = STLController(unique_identifier = agent_id, 
                                              dynamical_model   = model) 
        
    # Collect all the tasks for each agent.
    tasks_per_agent = {i:[] for i in multi_agent_system.agents_states.keys()}
    for i,j in multi_agent_system.task_graph.edges:
        print(i,j)
        stl_tasks = multi_agent_system.task_graph.task_list_for_edge(i,j)
        
        
        if i!=j :
            tasks_per_agent[i] += stl_tasks
            tasks_per_agent[j] += stl_tasks
        else :
            tasks_per_agent[i] += stl_tasks
    
    # Run token passing algorithm.
    tokens_sets = token_passing_algorithm(multi_agent_system.task_graph)
    # exchange the callbacks among agents
    for agent_id, Ti in tokens_sets.items():
        for agent_j,token in Ti.items() :
            if token == LeadershipToken.LEADER :
                controllers[agent_j].register("worse_impact",controllers[agent_id].on_worse_impact)
            elif token == LeadershipToken.FOLLOWER :
                controllers[agent_j].register("best_impact",controllers[agent_id].on_best_impact)
            else :
                raise ValueError("The token must be either LEADER or FOLLOWER. The presence of a token of type UNDEFINED is a bug. Contact the developers.")
            
    # Initialize the controllers.
    for agent_id,controller in controllers.items():
        controller.setup_optimizer(initial_conditions = multi_agent_system.agents_states,
                                   leadership_tokens = tokens_sets[agent_id],
                                   stl_tasks         = tasks_per_agent[agent_id],
                                   initial_time      = multi_agent_system.current_time)
    
    

    # Start simulation (sequential).
    for _ in tqdm(range(number_of_steps)):
        
        for agent_id in multi_agent_system.agents_models.keys():
                    state_history[agent_id] += [multi_agent_system.agents_states[agent_id].flatten()]
         
        try :
            logger.info(f"New round")
            for rounds in range(int(nx.diameter(multi_agent_system.task_graph)/2)+1):
                
                for agent_id,controller in controllers.items():
                    controller.compute_gamma_tilde_values(current_time = multi_agent_system.current_time, 
                                                        agents_state = multi_agent_system.agents_states)
                
                for agent_id,controller in controllers.items():
                    if controller.is_ready_to_compute_gamma:
                        controller.compute_gamma()
                        controller.compute_best_impact_for_follower(agents_states=multi_agent_system.agents_states,
                                                                    current_time = multi_agent_system.current_time)
                        controller.compute_worse_impact_for_leaders()
        except Exception as e:
            print(format_exc())
            break   
        
        inputs_per_agent = dict()
        # after this deterministic number of rounds everyone has all the information required to compute its control input
        for agent_id,controller in controllers.items():
            inputs_per_agent[agent_id] = controller.compute_control_input(current_states = multi_agent_system.agents_states, 
                                                                        current_time   = multi_agent_system.current_time)
            
        for agent_id,model in multi_agent_system.agents_models.items():
            multi_agent_system.agents_states[agent_id] = model.step_fun(multi_agent_system.agents_states[agent_id],inputs_per_agent[agent_id]).full()
        multi_agent_system.current_time += time_step
    
    
    fig, ax = plt.subplots()
    counter = 1
    for agent_id in multi_agent_system.agents_states.keys():
        state = np.vstack(state_history[agent_id])
        
        if counter == 1 :
            ax.scatter(state[0,0],state[0,1],marker="o",color="green", label="Initial position")
            ax.scatter(state[-1,0],state[-1,1],marker="o",color="red", label = "Final position")
        else :
            ax.scatter(state[0,0],state[0,1],marker="o",color="green")
            ax.scatter(state[-1,0],state[-1,1],marker="o",color="red")
        ax.plot(state[:,0],state[:,1],label=f"Agent {agent_id}")
        
        counter = 0
        
    ax.legend()
    plt.show()
    
    
    

# def visualize_gamma(gamma_history: dict[int,TimeSeries],exclude_list:list[int] = []) -> plt.Axes:
#     # now we can plot the results


#     fig, ax = plt.subplots()
    
#     for agent_id,history in gamma_history.items() :
        
#         time = np.array(list(history.keys()))
#         hist = np.array(list(history.values()))
#         if not (agent_id in exclude_list) :
#             ax.plot(time,hist,marker="s", label=fr"$\gamma_{{{agent_id}}}$")
    
    
    
    
#     ax.set_xlabel(r"$time [s]$")
#     ax.set_ylabel(r"$\gamma_i^k$")
#     ax.grid()
#     ax.legend()
    
    
#     return ax
    
    

# def visualize_barriers(agents_dict:dict[int,Agent],state_history:dict[int,TimeSeries],single_plot:bool = False,legend:bool =True)-> plt.Axes|None :

#     n_cols = 3
#     n_agents = len(agents_dict)
#     fig,ax = plt.subplots(-(-n_agents//n_cols),n_cols)
#     ax = ax.flatten()
#     counter = 0
#     if not single_plot :
#         #plot barriers
#         for agent_id,agent in agents_dict.items():
            
#             barriers : list[BarrierFunction] = agent.controller.barrier_functions
#             for barrier in barriers :
#                 barrier_history = TimeSeries()
#                 time_range = list(state_history[agent_id].keys())
#                 switch = barrier.switch_function
                
#                 for time in time_range :
#                     contributing_agents = barrier.contributing_agents
#                     try :
#                         function_inputs         = { f"state_{id}":state_history[id][time] for id in contributing_agents}
#                         function_inputs["time"] = time
#                         barrier_history[time]   = float(barrier.function.call(function_inputs)["value"]*switch(time))
#                     except KeyError:
#                         pass
                
#                 ax[counter].set_xlabel(r"$ time [s]$ ")
#                 ax[counter].set_ylabel("barrier")
#                 ax[counter].plot(list( barrier_history.keys()),list( barrier_history.values()),marker = 's', label=f"b_{contributing_agents }")

#             if legend :
#                 ax[counter].legend()
#             ax[counter].set_title("Agent "+str(agent_id))
#             ax[counter].grid()
            
#             counter +=1
    
#     else :
#         fig,ax = plt.subplots()
#         edges = set()
#         for agent_id,agent in agents_dict.items():
            
#             barriers : list[BarrierFunction] = agent.controller.barrier_functions
#             for barrier in barriers :
#                 barrier_history = TimeSeries()
#                 time_range = list(state_history[agent_id].keys())
#                 switch = barrier.switch_function

                
#                 contributing_agents = barrier.contributing_agents # it is always two in this case. one for the agent and one for the neighbour
                
#                 if len(contributing_agents) == 2 :
#                     if agent_id == contributing_agents[0] :
#                         neighbour_id = contributing_agents[1]
#                     else :
#                         neighbour_id = contributing_agents[0]
#                 else :
#                     neighbour_id = agent_id
                    
#                 if not ((agent_id,neighbour_id) in edges) : # to not repeat same edge
#                     try :
#                         for time in time_range :
#                             function_inputs         = { f"state_{id}":state_history[id][time] for id in contributing_agents}
#                             function_inputs["time"] = time
#                             barrier_history[time]   = float(barrier.function.call(function_inputs)["value"]*switch(time))
#                     except KeyError:
#                         pass
                    
#                     edges.update([(agent_id,neighbour_id),(neighbour_id,agent_id)])
                
                
#                     label = rf"$\phi_{{{agent_id,neighbour_id}}}$"
#                     ax.plot(list( barrier_history.keys()),list( barrier_history.values()), marker = 's',label=label)


#         if legend :
#             ax.legend()
#         ax.grid()
#         ax.set_xlabel(r"$time [s]$")
#         ax.set_ylabel(r"$b^{\phi}_{ij}(x^k,t^k) \qquad b^{\phi}_{i}(x^k,t) $")
#         return ax
                
            
        
    
#     fig,ax = plt.subplots(-(-n_agents//n_cols),n_cols)
#     ax = ax.flatten()
#     counter = 0 
#     # plot alpha functions
#     for agent_id,agent in agents_dict.items():
        
#         for barrier in agent.controller.barrier_functions :
#             alpha_barrier_history = TimeSeries()
#             time_range = list(state_history[agent_id].keys())
            
#             for time in time_range :
#                 contributing_agents = barrier.contributing_agents
#                 try :
#                     function_inputs         = { f"state_{id}":state_history[id][time] for id in contributing_agents}
#                     function_inputs["time"] = time
#                     alpha  = barrier.associated_alpha_function
#                     alpha_barrier_history[time]   =  float(alpha(barrier.function.call(function_inputs)["value"]))
                
#                 except KeyError:
#                     pass
            
#             ax[counter].set_xlabel(r"$time [s]$")
#             ax[counter].set_ylabel("alpha(b(x,t))")
#             ax[counter].plot(list(alpha_barrier_history.keys()),list( alpha_barrier_history.values()), label=f"b_{contributing_agents }")

        
#         ax[counter].legend()
#         ax[counter].set_title("Agent "+str(agent_id))
#         ax[counter].grid()
        
#         counter +=1
    
#     fig,ax = plt.subplots(-(-n_agents//n_cols),n_cols)
#     ax = ax.flatten()
#     counter = 0
#     # time derivative of the alpha functions
#     for agent_id,agent in agents_dict.items():
        
#         for barrier in agent.controller.barrier_functions :
#             db_dt = TimeSeries()
#             time_range = list(state_history[agent_id].keys())
            
#             for time in time_range :
#                 contributing_agents = barrier.contributing_agents
#                 try :
#                     function_inputs         = { f"state_{id}":state_history[id][time] for id in contributing_agents}
#                     function_inputs["time"] = time
#                     time_derivative = barrier.partial_time_derivative
                                        
#                     db_dt[time]   = float(barrier.partial_time_derivative.call(function_inputs)["value"])
                
#                 except KeyError:
#                     pass
            
#             ax[counter].set_xlabel(r"$ time [s]$")
#             ax[counter].set_ylabel("db_dt")
#             ax[counter].plot(list( db_dt.keys()),list( db_dt.values()), label=f"b_{contributing_agents }")

        
#         ax[counter].legend()
#         ax[counter].set_title("Agent "+str(agent_id))
#         ax[counter].grid()
        
#         counter +=1
    
    
#     fig,ax = plt.subplots(-(-n_agents//n_cols),n_cols)
#     ax = ax.flatten()
#     counter = 0
#     # time derivative of the alpha functions
#     for agent_id,agent in agents_dict.items():
        
#         for barrier in agent.controller.barrier_functions :
#             db_dt = TimeSeries()
#             time_range = list(state_history[agent_id].keys())
            
#             for time in time_range :
#                 contributing_agents = barrier.contributing_agents
#                 try :
#                     function_inputs         = { f"state_{id}":state_history[id][time] for id in contributing_agents}
#                     function_inputs["time"] = time
#                     time_derivative = barrier.partial_time_derivative
                                        
#                     db_dt[time]   = float(time_derivative.call(function_inputs)["value"])
                
#                 except KeyError:
#                     pass
            
#             ax[counter].set_xlabel(r"$time [s]$")
#             ax[counter].set_ylabel("db_dt")
#             ax[counter].plot(list( db_dt.keys()),list( db_dt.values()), label=f"b_{contributing_agents }")

        
#         ax[counter].legend()
#         ax[counter].set_title("Agent "+str(agent_id))
#         ax[counter].grid()
        
#         counter +=1

    
    