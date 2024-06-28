import polytope as pc
import numpy as np

P = pc.box2poly([[0, 1], [0, 1]])

print(P)
print(P.contains(np.array([[0.5], [0.5]])))