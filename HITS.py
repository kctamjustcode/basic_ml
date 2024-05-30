import numpy as np

M=np.array([[0,1,1,0],[0,0,0,0],[0,0,0,1],[0,0,1,0]])

H=np.matmul(M,np.transpose(M))
A=np.matmul(np.transpose(M),M)


Ia=np.array([1,1,1,1])
for i in range(200):
    Ia=A.dot(Ia)
    Ia=4*Ia/np.sum(Ia)
print(Ia)


Ih=np.array([1,1,1,1])
for i in range(200):
    Ih=H.dot(Ih)
    Ih=4*Ih/np.sum(Ih)
print(Ih)

print(0.6*Ia+0.4*Ih)

# ans: C,A,B,D
