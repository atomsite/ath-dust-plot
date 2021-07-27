import sys
sys.path.append("../")
import athenaplotlib as apl
import numpy as np
import matplotlib.pyplot as plt

class Star:
  def __init__(self,mass,period,ecc):
    self.mass   = mass
    self.period = period*86400
    self.ecc    = ecc
    self.phaseoff = 0.0
    return

wr104 = [Star(10,245,0.06),Star(20,245,0.06)]

wr98a = [Star(10,566,0.0),Star(18,566,0.0)]

wr140 = [Star(14.9,2869,0.896),Star(35.9,2869,0.896)]

def genorbits(s):
  orbit = []
  for phase in np.arange(0.0,1.0,0.001):
    time = phase * s[0].period
    pos = apl.calcOrbit(s[0],s[1],time)
    orbit.append((pos[0][0],pos[0][1],pos[1][0],pos[1][1]))
  return orbit
    

wr140pos = genorbits(wr140)

print(np.shape(wr140pos[0]))

print(wr140pos[0][:])

plt.scatter(wr140pos[:][0],wr140pos[:][1])
plt.show()




