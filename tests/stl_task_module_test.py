from stlddec.stl_task import *
import matplotlib.pyplot as plt

def separate_line():
    print("Passed")
    print("-------------------------------------------------------")

show_figures = True

import matplotlib.pyplot as plt
# Test 1 : Normal form
print("Test 1 : Normal form")
A = np.array([[20,1],[1,-20]])
b = np.array([[1],[-1]])
A,b = normal_form(A,b)
print(A)
print(np.linalg.norm(A,axis=1))

separate_line()
# Test 2 : Regular 2D polytope
print("Test 2 : Regular 2D polytope")


dist   = 1
step   = 2*dist +1
center = np.array([[0],[0]])
max_hp = 10
fig,ax = plt.subplots()
ax.set_xlim(- step ,max_hp*step+dist)
ax.set_ylim(- step , step )
ax.aspect = 'equal'

for hp in range(3,10):
    A,z = regular_2D_polytope(hp,1)

    p = pc.Polytope(A,z+A@center)
    p.plot(alpha=  0.5,ax = ax)
    center = center + np.array([[step],[0]])
ax.set_title("Regular 2D Polytopes")

separate_line()
# Test 3 : Random 2D polytope
print("Test 3 : Random 2D polytope")
dist   = 1
step   = 2*dist +1
center = np.array([[0],[0]])
max_hp = 10
fig,ax = plt.subplots()
ax.set_xlim(- step ,max_hp*step+dist)
ax.set_ylim(- step , step )
ax.aspect = 'equal'

for hp in range(3,10):
    A,z = random_2D_polytope(hp,1)

    p = pc.Polytope(A,z+A@center)
    p.plot(alpha=  0.5,ax = ax)
    center = center + np.array([[step],[0]])

ax.set_title("Random 2D Polytopes")

separate_line()
# Test 4: Test temporal operators 
print("Test 4: Test temporal operators and time intervals")
time_interval1 = TimeInterval(0,10)
time_interval2 = TimeInterval(3,15)
intersection   = TimeInterval(3,10)

assert time_interval1/time_interval2 == intersection 

time_interval1 = TimeInterval(None,None)
time_interval2 = TimeInterval(3,15)
assert time_interval1/time_interval2 == TimeInterval(None,None) 
assert time_interval1.is_empty()

time_interval1 = TimeInterval(3,3)
time_interval2 = TimeInterval(3,15)
assert time_interval1/time_interval2 == TimeInterval(3,3) 
assert (time_interval1/time_interval1).is_singular()
separate_line()
print("Temporal operator G_[0,10]")
temporal_operator = AlwaysOperator(TimeInterval(0,10))
print("Always operator time of remotion: ",temporal_operator.time_of_remotion)
print("Always operator time of satisfaction: ",temporal_operator.time_of_satisfaction)
separate_line()

temporal_operator = EventuallyOperator(TimeInterval(0,10))
print("Eventually operator time of remotion: ",temporal_operator.time_of_remotion)
print("Eventually  operator time of satisfaction: ",temporal_operator.time_of_satisfaction)
print("Time of satisfaction is the same as time of remotion")
assert temporal_operator.time_of_satisfaction >= 0
assert temporal_operator.time_of_satisfaction <= 10
assert temporal_operator.time_of_remotion == temporal_operator.time_of_satisfaction
separate_line()


# Test 5: Test predicates
A,b = regular_2D_polytope(5,1)
P   = CollaborativePredicate(pc.Polytope(A,b.flatten()),source_agent_id=0,target_agent_id=1)    
# P   = AbstractPolytopicPredicate(pc.Polytope(A,b)) # throws an error
print(A)
assert P.is_parametric
assert P.state_space_dim == 2
assert P.num_hyperplanes == 5
assert P.num_vertices == 5

P_ind = IndependentPredicate(pc.Polytope(A,b),agent_id=0)
assert P_ind.is_parametric
P_collab = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=0,target_agent_id=1)
P_collab.flip()

assert P_collab.source_agent == 1
assert not np.all(np.linalg.norm(P_collab.A +A,axis=1))

separate_line()
# Test 6: Test tasks
print("Test 6: Test tasks")
task = StlTask(AlwaysOperator(TimeInterval(0,10)),P)
assert task.state_space_dimension == 2
assert task.is_parametric
assert task.predicate == P
assert task.task_id == 0

for jj in range(10):
    task = create_parametric_collaborative_task_from(task,source_agent_id=0,target_agent_id=1)
    assert task.task_id == jj+1
    print("New task created with id: ",task.task_id)

if show_figures:
    plt.show()

# Test inclusions 
print("Test 7: Test inclusion matrices")
A,b = regular_2D_polytope(5,1)
print(b)
P_including = CollaborativePredicate(pc.Polytope(A,b),source_agent_id=0,target_agent_id=1)
A,b = regular_2D_polytope(3,1)
P_included = IndependentPredicate(pc.Polytope(A,b),agent_id=0)

M,Z = get_M_and_Z_matrices_from_inclusion(P_including,P_included)
print("Matrix M: should be 15x3 and equal to  [A@[I v1] ,A@[I v2],A@[I v3]], with v_i vertices of the included polytope")
print(M)
print("Matrix Z: should be 15x3 and equal to  [[A z],[A,z],[A,z]]")
print(Z)
