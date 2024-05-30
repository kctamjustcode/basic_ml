import numpy as np
from numpy import linalg

w,v = linalg.eig(np.diag((1,2,3)))

print(w)
print(v)
print(v[1])

a = np.array([[1, 2], [3, 4]])
e = np.mean(a, axis=1)
print(e)

b=np.append(a,[[5,6]])
b=b.reshape(3,2)

print(a)
print(b)
