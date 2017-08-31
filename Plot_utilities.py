import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D



def plot(RLMCObj, title = "Optimal State Value Function"): #plot value function against state
   fig = plt.figure(figsize=(14,12))
   ha = fig.add_subplot(1,1,1, projection='3d')
   X = range(1,11)
   Y = range(1,22)
   X1, Y1 = np.meshgrid(X, Y)
   Z = RLMCObj.getVtable().T
   ha.plot_wireframe(X1, Y1, Z, rstride=1, cstride=1)
   ha.set_ylabel("Player's current sum")
   ha.set_xlabel("Dealer's starting card")
   ha.set_zlabel("Maximum State Value")
   plt.title(title)
   plt.show()



def MSE(Q1, Q2): #compute MSE between two Q tables
      assert Q1.size == Q2.size,  "Incompatible tables"
      return np.sum(np.square(Q1 - Q2))/ Q1.size



