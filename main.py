#!/usr/bin/python
#coding: utf-8
import subprocess, os, sys
import utils
import numpy as np
# import csv as libcsv
import ipdb
import matplotlib.pyplot as plt
# Use the CPU in 64-bit mode.
# from sknn.platform import cpu64
from sknn.platform import cpu32, threading, threads4
from sknn.mlp import Layer, Classifier
from sklearn.metrics import confusion_matrix
# from sklearn.neural_network import MLPClassifier
from unbalanced_dataset.over_sampling import SMOTE

#Save same output to file
# tee = subprocess.Popen(["tee", os.path.join('results', 'log.txt')], stdin=subprocess.PIPE)
# os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
# os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

#create needed folder
# if not os.path.exists('results'):
# 	os.makedirs('results')

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
	target = np.concatenate((classe0[:halfx,-1],classe1[:halfy,-1])).astype(int)

	print('>> Training')

	print('Classe 0 training size:',len(trainingsetx))
	print('Classe 1 training size:',len(trainingsety))

	'''OVERSAMPLING'''

#	Oversampling por SMOTE
	sm = SMOTE(kind='regular', verbose=True)
	balancedTrainingSetData,balancedTrainingSetTarget = sm.fit_transform(data, target)
	
#	Oversampling por repetição simples
	# balancedTrainingSetData = np.concatenate((trainingsetx,trainingsety.repeat(42, axis=0)))
	# balancedTrainingSetTarget = np.concatenate((classe0[:halfx,-1],classe1[:halfy,-1].repeat(42, axis=0))).astype(int)	

	print('Balanced training size:',len(balancedTrainingSetData))
	print('Balanced training data shape:',balancedTrainingSetData.shape)
	print('Balanced training target shape:',balancedTrainingSetTarget.shape)

# ######################################################################################

# 	#Primeiro quarto - conjunto de validação - 25%
	quarter1x = int((len(classe0)*.75))
	quarter1y = int((len(classe1)*.75))

	validationsetx = classe0[halfx:quarter1x,:-1]
	validationsety = classe1[halfy:quarter1y,:-1]

	data = np.concatenate((validationsetx,validationsety))
	target = np.concatenate((classe0[halfx:quarter1x,-1],classe1[halfy:quarter1y,-1])).astype(int)

	print('>> Validation size:',len(data))

	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular')
	balancedValidationSetData, balancedValidationSetTarget = sm.fit_transform(data, target)

#	Oversampling por repetição simples
	# balancedValidationSetData = np.concatenate((validationsetx,validationsety.repeat(42, axis=0)))
	# balancedValidationSetTarget = np.concatenate((classe0[halfx:quarter1x,-1],classe1[halfy:quarter1y,-1].repeat(42, axis=0))).astype(int)

	print('Balanced validation size:',len(balancedValidationSetData))
	print('Balanced validation data shape:',balancedValidationSetData.shape)
	print('Balanced validation target shape:',balancedValidationSetTarget.shape)

# #######################################################################################

# 	#Segundo quarto - conjunto de teste - 25%
	testsetx = classe0[quarter1x:,:-1]
	testsety = classe1[quarter1y:,:-1]

	data = np.concatenate((testsetx,testsety))
	target = np.concatenate((classe0[quarter1x:,-1],classe1[quarter1y:,-1])).astype(int)

	print('>> Test')
	print('Classe 0 test size:',len(testsetx))
	print('Classe 1 test size:',len(testsety))

	print('Balanced test size:',len(data))

	'''LEARNING'''
	layers = [
		Layer(type='Sigmoid',name="hidden_layer_1",units=3),
		Layer(type='Sigmoid',name="hidden_layer_2",units=2),
		Layer(type='Softmax',name="output_layer",units = 3)
	]


	error_train = []
	error_valid = []

	def store_errors(avg_valid_error, avg_train_error, **_):
		error_train.append(avg_valid_error)
		error_valid.append(avg_train_error)

	print('Initializing cliassifier')
	
	nn = Classifier(
	    layers=layers,
	    learning_rate=0.0001,
	    n_iter=50,
	    valid_set=(balancedValidationSetData,balancedValidationSetTarget),
	    callback={'on_epoch_finish': store_errors},
	    verbose = True
	    )
	
	print('Fitting')
	ipdb.set_trace()
	nn.fit(balancedTrainingSetData,balancedTrainingSetTarget)
	#score = nn.score(balancedValidationSetData,balancedValidationSetTarget)

	print('Testing')
	errors = 0

	print(data)
	predictions = np.squeeze(np.asarray(nn.predict(data)))

	for predicted,obj in zip(predictions,target):

		result = predicted

		print(result, obj,end='')
		if result != obj:
			print(' error')
			errors += 1
		print()

	# plt.plot(error_train, error_valid)
	# #utils.save('test')
	# plt.show()

	utils.plot_confusion_matrix(confusion_matrix(target, predictions))
	utils.plot_mse_curve(np.array(error_train), np.array(error_valid))
	utils.plot_roc_curve(target, predictions)

	print("acurracy:", ((len(data)-errors)/len(data))*100,'%')
	print('errors',errors,'of', len(data))


# 	# clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1, shuffle=True)
# 	# clf.fit(classe0, classe1)