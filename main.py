import numpy as np
import csv as libcsv
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
	    	# data.append(line_[:-1])
	    	# target.append(int(line_[-1]))

	    	#for separeted data
	    	if line_[-1] == 0:
	    		classe0.append(line_[:-1])
	    	elif line_[-1] == 1:
	    		classe1.append(line_[:-1])

	classe0 = np.array(classe0)
	classe1 = np.array(classe1)
	# data = np.array(data)
	# target = np.array(target)

	print('Classe 0 size:',len(classe0))
	print('Classe 1 size:',len(classe1))

	#randomize classe 0
	np.random.shuffle(classe0)

    #randomize classe 1
	np.random.shuffle(classe1)

	#Preparando conjunto de treinamento - 50%
	halfx = int((len(classe0)*.5))
	halfy = int((len(classe1)*.5))

	trainingsetx = classe0[0:halfx]
	trainingsety = classe1[0:halfy]

	print('Classe 0 training size:',len(trainingsetx))
	print('Classe 1 training size:',len(trainingsety))

	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular', verbose=True)
	svmx, svmy = sm.fit_transform(trainingsetx, trainingsety)

	print('Classe 0 balanced training size:',len(svmx))
	print('Classe 1 balanced training size:',len(svmy))

######################################################################################

	#Primeiro quarto - conjunto de validação - 25%
	quarter1x = int((len(classe0)*.75))
	quarter1y = int((len(classe1)*.75))

	validationsetx = classe0[halfx:quarter1x]
	validationsety = classe1[halfy:quarter1y]

	print('Classe 0 validation size:',len(validationsetx))
	print('Classe 1 validation size:',len(validationsety))

	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular', verbose=True)
	svmx, svmy = sm.fit_transform(validationsetx, validationsety)	

	print('Classe 0 balanced validation size:',len(svmx))
	print('Classe 1 balanced validation size:',len(svmy))	

#######################################################################################

	#Segundo quarto - conjunto de teste - 25%
	testsetx = classe0[quarter1x:]
	testsety = classe1[quarter1y:]

	print('Classe 0 test size:',len(testsetx))
	print('Classe 1 test size:',len(testsety))

	'''OVERSAMPLING'''

	sm = SMOTE(kind='regular', verbose=True)
	svmx, svmy = sm.fit_transform(validationsetx, validationsety)	

	print('Classe 0 balanced validation size:',len(svmx))
	print('Classe 1 balanced validation size:',len(svmy))


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
