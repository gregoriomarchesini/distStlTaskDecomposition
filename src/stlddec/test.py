
import numpy as np

class SomeClass():
    counter = 0
    
    def __init__(self):
        self._id = SomeClass.counter
        SomeClass.counter += 1

A = SomeClass() # 1
B = SomeClass() # 1
C = SomeClass() # 1
print(A._id) # 1
print(B._id) # 2
print(C._id) # 3

def tuple_to_int(t:tuple) -> int :
    """Converts a tuple to an integer"""
    return int("".join(str(i) for i in t))


print(tuple_to_int((1,2))) # 12
