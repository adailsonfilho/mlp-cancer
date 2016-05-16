import numpy as np
import ipdb

# Use the CPU in 64-bit mode.
# from sknn.platform import cpu64
from sknn.mlp import Layer, Classifier
# from sklearn.neural_network import MLPClassifier
from unbalanced_dataset.over_sampling import SMOTE

if __name__ == '__main__':

	'''PRE PROCESSING'''

	#READ
	classe0 = []
	classe1 = []
	data = []
	target =[]

	with open('mammography-consolidated.csv', 'r') as csv:
	    for line in csv:
	    	line_ = [ float(val) for val in line.split(',')]

	    	#for united data
	    	data.append(line_[:-1])
	    	target.append(int(line_[-1]))

	    	#for separeted data
	    	if line_[-1] == 0:
	    		classe0.append(line_[:-1])
	    	elif line_[-1] == 1:
	    		classe1.append(line_[:-1])

	classe0 = np.array(classe0)
	classe1 = np.array(classe1)
	data = np.array(data)
	target = np.array(target)

	print('Classe 0 size:',len(classe0))
	print('Classe 1 size:',len(classe1))

	'''LEARNING'''

	# hiddenlayer1 = mlp.Layer(type='Sigmoid',name="hidden_layer_1",units=4)
	# hiddenlayer2 = mlp.Layer(type='Sigmoid',name="hidden_layer_2",units=3)
	# outputlayer = mlp.Layer(type='Linear',name="output_layer",units=1)

	# nn = Classifier(
	#     layers=[
	#     	Layer(type='Sigmoid',name="hidden_layer_1",units=4),
	# 		Layer(type='Sigmoid',name="hidden_layer_2",units=4),
	# 		Layer(type='Softmax',name="output_layer",units=1)],
	#     learning_rate=0.001,
	#     n_iter=100)

	# nn.fit(data,target)

	# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1, shuffle=True)
	# clf.fit(classe0, classe1)

	sm = SMOTE(kind='regular', verbose='verbose')
	svmx, svmy = sm.fit_transform(data, target)

	print("Tamanho da classe 0",len(svmx))
	print("Tamanho da classe 1",len(svmy))

    # print('SMOTE bordeline 1')
    # sm = SMOTE(kind='borderline1', verbose=verbose)
    # svmx, svmy = sm.fit_transform(classe0, classe1)

    # print('SMOTE bordeline 2')
    # sm = SMOTE(kind='borderline2', verbose=verbose)
    # svmx, svmy = sm.fit_transform(classe0, classe1)

    # print('SMOTE SVM')
    # svm_args={'class_weight': 'auto'}
    # sm = SMOTE(kind='svm', verbose=verbose, **svm_args)
    # svmx, svmy = sm.fit_transform(classe0, classe1)
