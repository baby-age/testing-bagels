import pandas as pand
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = gen.generate_graphs(300, 40, 100)
newdata = []
for i in range(len(data.index)):
	newdata.append([list(emb.embed(data.X[i])), data.y[i]])

data = pand.DataFrame(newdata, columns = ['X', 'y'])

x1 = []
x2 = []
x3 = []
for i in range(len(data.index)):
	x1.append(data['X'][i][0])
	x2.append(data['X'][i][1])
	x3.append(data['X'][i][2])

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x1, x2, x3, c = data['y'])
ax.set_xlabel('Ave. dist.')
ax.set_ylabel('Max. eig.gap')
ax.set_zlabel('sum eig.val')
plt.show()
