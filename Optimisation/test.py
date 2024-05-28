import numpy as np

x = np.arange(500)#t
y = np.arange(5)#Ps
M = np.outer(x.T,x)

print(M)

np.linalg.inv(M)