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

	
	'''OVERSAMPLING'''
	
	sm = SMOTE(kind='regular', verbose='verbose')
	svmx, svmy = sm.fit_transform(data, target)

	print("Dados balanceados")
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

    '''SPLITING DATA'''

    #rondomize classe 0
    np.random.shuffle(svmx)

    #rondomize classe 1
    np.random.shuffle(svmy)

    #Prieira metade - 50%
    halfx = int((len(svmx)*.5))
    halfy = int((len(svmy)*.5))
    trainingsetx = svmx[0:halfx]
    trainingsety = svmy[0:halfy]

    #Prieiro quarto - 25%
    quarter1x = int((len(svmy)*.75))
    quarter1y = int((len(svmy)*.75))
    validationsetx = svmx[halfx:quarter1x]
    validationsety = svmy[halfy:quarter1y]

    #Segundo quarto - 25%
    testsetx = svmx[quarter1x:]
    testsety = svmy[quarter1y:]


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
