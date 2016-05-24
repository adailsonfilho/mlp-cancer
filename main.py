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

	with open('mammography-consolidated.csv', 'r') as csv:
	    for line in csv:
	    	line_ = [ float(val) for val in line.split(',')]

	    	#for separeted data
	    	if line_[-1] == 0:
	    		classe0.append(line_)
	    	elif line_[-1] == 1:
	    		classe1.append(line_)

	classe0 = np.array(classe0)
	classe1 = np.array(classe1)

	print('Classe 0 size:',len(classe0))
	print('Classe 1 size:',len(classe1))

	#randomize classe 0
	np.random.shuffle(classe0)

    #randomize classe 1
	np.random.shuffle(classe1)

	#Preparando conjunto de treinamento - 50%
	halfx = int((len(classe0)*.5))
	halfy = int((len(classe1)*.5))

	trainingsetx = classe0[:halfx, :-1]
	trainingsety = classe1[:halfy, :-1]

	data = np.concatenate((trainingsetx,trainingsety))
	target = np.concatenate((classe0[:halfx,-1],classe1[:halfy,-1]))

	print('Classe 0 training size:',len(trainingsetx))
	print('Classe 1 training size:',len(trainingsety))

	'''OVERSAMPLING'''

	sm = SMOTE(kind='regular', verbose=True)
	balancedTrainingSetData,balancedTrainingSetTarget = sm.fit_transform(data, target)

	print('Balanced training size:',len(balancedTrainingSetData))
	print('Balanced training data shape:',balancedTrainingSetData.shape)
	print('Balanced training target shape:',balancedTrainingSetTarget.shape)

######################################################################################

	#Primeiro quarto - conjunto de validação - 25%
	quarter1x = int((len(classe0)*.75))
	quarter1y = int((len(classe1)*.75))

	validationsetx = classe0[halfx:quarter1x,:-1]
	validationsety = classe1[halfy:quarter1y,:-1]

	data = np.concatenate((validationsetx,validationsety))
	target = np.concatenate((classe0[halfx:quarter1x,-1],classe1[halfy:quarter1y,-1]))

	print('Validation size:',len(validationsetx))

	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular', verbose=True)
	balancedValidationSetData, balancedValidationSetTarget = sm.fit_transform(data, target)

	print('Balanced validation size:',len(balancedValidationSetData))

#######################################################################################

	#Segundo quarto - conjunto de teste - 25%
	testsetx = classe0[quarter1x:]
	testsety = classe1[quarter1y:]

	data = np.concatenate((testsetx,testsety))
	target = np.concatenate((classe0[quarter1x:,-1],classe1[quarter1y:,-1]))

	print('Classe 0 test size:',len(testsetx))
	print('Classe 1 test size:',len(testsety))

	'''OVERSAMPLING'''

	sm = SMOTE(kind='regular', verbose=True)
	balancedTestSetData, balancedTestSetTarget = sm.fit_transform(data, target)

	print('Balanced test size:',len(balancedTestSetData))

	'''LEARNING'''
	inputlayer = Layer(type='Sigmoid',name="input_layer_1",units=6)
	hiddenlayer1 = Layer(type='Sigmoid',name="hidden_layer_1",units=4)
	outputlayer = Layer(type='Softmax',name="output_layer",units=2)

	nn = Classifier(
	    layers=[
	    	inputlayer,
	    	hiddenlayer1,
			outputlayer],
	    learning_rate=0.001,
	    n_iter=100)

	nn.fit(balancedTrainingSetData,balancedTrainingSetTarget)

	# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1, shuffle=True)
	# clf.fit(classe0, classe1)