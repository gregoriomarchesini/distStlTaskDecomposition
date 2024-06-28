import numpy as np

A = np.eye(3)
print([np.expand_dims(vertex,1) for vertex in [*A]])