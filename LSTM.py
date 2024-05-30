import math
import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))


# part (a): LSTM
wf=np.array([0.7,0.4,0.1])
bf=0.1
wi=np.array([0.2,0.6,0.7])
bi=0.4
wa=np.array([0.3,0.2,0.1])
ba=0.3
wo=np.array([0.6,0.3,0.1])
bo=0.2

s0=0
y0=0

y_t1=0.2
y_t2=0.4

f1=sigmoid(np.dot(wf,np.array([0.3,0.6,y0])) + bf)
i1=sigmoid(np.dot(wi,np.array([0.3,0.6,y0])) + bi)
a1=math.tanh(np.dot(wa,np.array([0.3,0.6,y0])) + ba)
s1=f1*s0+i1*a1
o1=sigmoid(np.dot(wo,np.array([0.3,0.6,y0]))+bo)
y1=o1*math.tanh(s1)

f2=sigmoid(np.dot(wf,np.array([0.1,1.0,y1])) + bf)
i2=sigmoid(np.dot(wi,np.array([0.1,1.0,y1])) + bi)
a2=math.tanh(np.dot(wa,np.array([0.1,1.0,y1])) + ba)
s2=f2*s1+i2*a2
o2=sigmoid(np.dot(wo,np.array([0.1,1.0,y1]))+bo)
y2=o2*math.tanh(s2)

print("f1="+str(round(f1,4)))
print("i1="+str(round(i1,4)))
print("a1="+str(round(a1,4)))
print("s1="+str(round(s1,4)))
print("o1="+str(round(o1,4)))

print("f2="+str(round(f2,4)))
print("i2="+str(round(i2,4)))
print("a2="+str(round(a2,4)))
print("s2="+str(round(s2,4)))
print("o2="+str(round(o2,4)))

print("the error, y_t1-y1, is "+ str(round(y_t1-y1,4)))
print("the error, y_t2-y2, is "+ str(round(y_t2-y2,4)))
