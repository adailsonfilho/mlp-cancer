import numpy as np

# Use the CPU in 64-bit mode.
# from sknn.platform import cpu64
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':

	'''PRE PROCESSING'''

	#READ
	classe0 = []
	classe1 = []

	with open('mammography-consolidated.csv', 'r') as csv:
	    for line in csv:
	    	line_ = [ float(val) for val in line.split(',')]
	    	if line_[-1] == 0:
	    		classe0.append(line_[:-1])
	    	elif line_[-1] == 1:
	    		classe1.append(line_[:-1])

	classe0 = np.array(classe0)
	classe1 = np.array(classe1)

	print('Classe 0 size:',len(classe0))
	print('Classe 1 size:',len(classe1))

	'''LEARNING'''

	# hiddenlayer1 = mlp.Layer(type='Sigmoid',name="hidden_layer_1",units=4)
	# hiddenlayer2 = mlp.Layer(type='Sigmoid',name="hidden_layer_2",units=3)
	# outputlayer = mlp.Layer(type='Linear',name="output_layer",units=1)

	# nn = Classifier(
	#     layers=[
	#     	Layer(type='Sigmoid',name="hidden_layer_1",units=4),
	# 		Layer(type='Sigmoid',name="hidden_layer_2",units=3),
	# 		Layer(type='Linear',name="output_layer",units=1)],
	#     learning_rate=0.001,
	#     n_iter=100)

	# nn.fit(classe0, classe1)

	clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1, shuffle=True)
	clf.fit(classe0, classe1)
