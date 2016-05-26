#!/usr/bin/python
#coding: utf-8
import numpy as np
# import csv as libcsv
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

##################################################################################

	#Preparando conjunto de treinamento - 50%
	halfx = int((len(classe0)*.5))
	halfy = int((len(classe1)*.5))

	trainingsetx = classe0[:halfx, :-1]
	trainingsety = classe1[:halfy, :-1]

	data = np.concatenate((trainingsetx,trainingsety))
	target = np.concatenate((classe0[:halfx,-1],classe1[:halfy,-1]))

	print('>> Training')

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

	vdata = np.concatenate((validationsetx,validationsety))
	vtarget = np.concatenate((classe0[halfx:quarter1x,-1],classe1[halfy:quarter1y,-1]))

	print('>> Validation size:',len(validationsetx))

	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular')
	balancedValidationSetData, balancedValidationSetTarget = sm.fit_transform(vdata, vtarget)

	print('Balanced validation size:',len(balancedValidationSetData))
	print('Balanced validation data shape:',balancedValidationSetData.shape)
	print('Balanced validation target shape:',balancedValidationSetTarget.shape)

#######################################################################################

	#Segundo quarto - conjunto de teste - 25%
	testsetx = classe0[quarter1x:,:-1]
	testsety = classe1[quarter1y:,:-1]

	tdata = np.concatenate((testsetx,testsety))
	ttarget = np.concatenate((classe0[quarter1x:,-1],classe1[quarter1y:,-1]))

	print('>> Test')
	print('Classe 0 test size:',len(testsetx))
	print('Classe 1 test size:',len(testsety))

	balancedTestSetData = tdata
	balancedTestSetTarget = ttarget

	print('Balanced test size:',len(balancedTestSetData))

	'''LEARNING'''
	hiddenlayer1 = Layer(type='Sigmoid',name="input_layer_1",units=4)
	hiddenlayer2 = Layer(type='Sigmoid',name="hidden_layer_1",units=3)
	outputlayer = Layer(type='Softmax',name="output_layer")

	print('Initializing cliassifier')
	nn = Classifier(
	    layers=[
	    	# hiddenlayer1,
	    	hiddenlayer2,
			outputlayer],
	    learning_rate=0.001,
	    n_iter=20,
	    valid_set=(balancedValidationSetData,balancedValidationSetTarget),
	    verbose = True
	    )

	print('Fitting')
	nn.fit(balancedTrainingSetData,balancedTrainingSetTarget)

	print('Testing')
	errors = 0
	for sample,obj in zip(data,target):
		 if nn.predict(sample) != obj:
		 	errors+= 1

	print("acurracy:", ((len(data)-errors)/len(data))*100,'%')
	print('errors',errors,'of', len(data))


	# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1, shuffle=True)
	# clf.fit(classe0, classe1)