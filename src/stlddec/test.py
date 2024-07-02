
import numpy as np

b = np.ones((2,1))
A = 3*np.eye(3)

print(np.kron(b,A))