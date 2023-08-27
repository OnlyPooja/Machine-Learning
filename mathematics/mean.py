import numpy as np
import statistics as states
height=[170,130,125,70,111]
mean=np.mean(height)
states=states.mode(height)
variance=np.var(height)
st=np.std(height)
height.sort()
median=np.median(height);
print("mean",mean)
print("median",median)
print("mode",states)
print("variance",variance)
print("stand.deviation",st)