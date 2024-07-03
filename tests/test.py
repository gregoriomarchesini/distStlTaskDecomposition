import casadi as ca
import numpy as np

a = np.eye(3)
print(a)
a = ca.DM(a)
print(a)