# EDGE 32
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(5,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([5,-5]),
                                           source_agent_id=3,
                                           target_agent_id=2)
task  = pmod.G(5,10) >> predicate
task_graph.attach(task)



# EDGE 32
# Create aa regular polytope.
polytope     = pmod.regular_2D_polytope(3,2)
predicate    = pmod.CollaborativePredicate( polytope_0=  polytope,
                                           center=np.array([1,0]),
                                           source_agent_id=3,
                                           target_agent_id=2)
task  = pmod.F(20,30) >> predicate
task_graph.attach(task)

