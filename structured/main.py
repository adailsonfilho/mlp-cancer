import numpy as np
from preprocess import Data
from config import Config
from oversampler import Oversampler
from enums import Oversampling
from sknn.platform import cpu32, threading, threads4
from sknn.mlp import Layer, Classifier


if __name__ == '__main__':

	verbose = True

	"""
	Pre-process
	"""

	base = Data('..\mammography-consolidated.csv',verbose=verbose)
	training, validation, testing = base.split()

	"""
	Setup experiment config
	"""

	#currently, only oversample
	sampling_options = [Oversampling.Repeat, Oversampling.SmoteRegular, Oversampling.DontUse]

	# learning_rule = stochastic gradient descent ('sgd'), 'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'
	learning_rule_options = ['sgd', 'momentum','rmsprop']

	#following the SKNN docs
	activation_function_options = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

	#based on W.I.W.T. - What I Want To
	topology_options = [
		[
			{'name':'hidden', 'units':5}
		],
		[
			{'name':'hidden', 'units':20}
		],
		[
			{'name':'hidden1', 'units':5},
			{'name':'hidden2', 'units':2},
		],
		[
			{'name':'hidden1', 'units':5},
			{'name':'hidden2', 'units':4},
			{'name':'hidden3', 'units':3}
		],
		[
			{'name':'hidden1', 'units':50},
			{'name':'hidden2', 'units':30},
		]

	]

	for opt_samp in sampling_options:

		if opt_samp != Oversampling.DontUse:
			"""
			TRAINING OVER SAMPLE
			"""
			oversampler = Oversampler(opt_samp,training['data'], training['target'],True )
			training['data'], training['target'] = oversampler.balance()

			"""
			TRAINING OVER SAMPLE
			"""
			oversampler = Oversampler(opt_samp,validation['data'], validation['target'],True )
			validation['data'], validation['target'] = oversampler.balance()

			"""
			DO NOT MAKE SENSE OVERSAMPLING OF TESTING SET
			"""
		base = {'training':training, 'validation': validation, 'testing': testing}

		for opt_learning in learning_rule_options:
			for opt_top in topology_options:
				for opt_actvfunc in activation_function_options:

					config = Config( base, opt_learning, opt_top, opt_actvfunc, force_overfiting = False)

					#data storing for charts
					error_train = []
					error_valid = []

					def store_errors(avg_valid_error, avg_train_error, **_):
						error_train.append(avg_valid_error)
						error_valid.append(avg_train_error)

					'''LEARNING'''
					#setting up Neural Network

					layers = [ Layer(type=opt_actvfunc,name=topology['name'],units=topology['units']) for topology in opt_top];
					layers.append(Layer(type='Softmax',name="output_layer"))

					nn = Classifier(
				    layers=layers,
				    learning_rate=0.0001,
				    n_iter=50,
				    valid_set=(base['validation']['data'],base['validation']['target']),
				    callback={'on_epoch_finish': store_errors},
				    verbose = verbose
				    )

					nn.fit(base['training']['data'],base['training']['target'])

					print('Testing')
					errors = 0

					print(data)
					predictions = np.squeeze(np.asarray(nn.predict(data)))

					for predicted,obj in zip(predictions,target):
						result = predicted

						if result != obj:
							# print(' error')
							errors += 1

					print("acurracy:", ((len(data)-errors)/len(data))*100,'%')
					print('errors',errors,'of', len(data))