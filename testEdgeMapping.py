from src.decomposition_module import edgeMapping



a = edgeMapping()
a[(1,2)] = 3
a[(2,1)] = 2
print((2,1) in a)
print(a)
