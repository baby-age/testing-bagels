import pandas as pand
import numpy as np
import testing_bagels.embedder as emb
import testing_bagels.graphgen as gen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

train_data = gen.generate_graphs(300, 40, 100)
test_data = gen.generate_graphs(100, 40, 100)

new_train_data = []
new_test_data = []
for i in range(len(train_data.index)):
	new_train_data.append([list(emb.embed(train_data.X[i])), train_data.y[i]])

for i in range(len(test_data.index)):
	new_test_data.append([list(emb.embed(test_data.X[i])), test_data.y[i]])

train_data = pand.DataFrame(new_train_data, columns = ['X', 'y'])
test_data = pand.DataFrame(new_test_data, columns = ['X', 'y'])

train_x1 = []
train_x2 = []
train_x3 = []
for i in range(len(train_data.index)):
	train_x1.append(train_data['X'][i][0])
	train_x2.append(train_data['X'][i][1])
	train_x3.append(train_data['X'][i][2])

test_x1 = []
test_x2 = []
test_x3 = []
for i in range(len(test_data.index)):
	test_x1.append(test_data['X'][i][0])
	test_x2.append(test_data['X'][i][1])
	test_x3.append(test_data['X'][i][2])

clf = linear_model.Lasso(alpha=0.1)
clf.fit(np.column_stack((train_x1, train_x2, train_x3)), train_data['y'])
pred = clf.predict(np.column_stack((test_x1, test_x2, test_x3)))
print("Average error: ", sum(pred-test_data['y'])/len(test_data))

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(train_x1, train_x2, train_x3, c = train_data['y'])
ax.set_xlabel('Ave. dist.')
ax.set_ylabel('Max. eig.gap')
ax.set_zlabel('sum eig.val')
plt.show()
