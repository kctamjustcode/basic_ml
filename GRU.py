import math
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))


# part (b): GRU
wr=np.array([0.5,0.2,0.3])
br=0.2
wa=np.array([0.2,0.4,0.1])
ba=0.1
wu=np.array([0.1,0.3,0.2])
bu=0.1

y0=0

y_t1=0.2
y_t2=0.4


r1=sigmoid(np.dot(wr,np.array([0.3,0.6,y0])) + br)
a1=math.tanh(np.dot(wa,np.array([0.3,0.6,r1*y0])) + ba) # missing r_0 here
u1=sigmoid(np.dot(wu,np.array([0.3,0.6,y0]))+bu)
y1=(1-u1)*y0+u1*a1

r2=sigmoid(np.dot(wr,np.array([0.1,1.0,y1])) + br)
a2=math.tanh(np.dot(wa,np.array([0.1,1.0,r2*y1])) + ba) # missing r_1 here
u2=sigmoid(np.dot(wu,np.array([0.1,1.0,y1]))+bu)
y2=(1-u2)*y1+u2*a2

print("r1="+str(round(r1,4)))
print("a1="+str(round(a1,4)))
print("u1="+str(round(u1,4)))
print("y1="+str(round(y1,4)))

print("r2="+str(round(r2,4)))
print("a2="+str(round(a2,4)))
print("u2="+str(round(u2,4)))
print("y2="+str(round(y2,4)))

print("the errors, y_t1-y1, is: "+str(round(y_t1-y1,4)))
print("the errors, y_t2-y2, is: "+str(round(y_t2-y2,4)))
