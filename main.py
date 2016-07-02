import numpy as np
from preprocess import Data
from config import Config
from oversampler import Oversampler
from enums import Oversampling
from sknn.platform import cpu32, threading, threads4
from sknn.mlp import Layer, Classifier
import datetime
import os, sys
import math

from metrics import Metrics

if __name__ == '__main__':

	verbose = True

	"""
	Pre-process
	"""

	base = Data('mammography-consolidated.csv', verbose=verbose)
	training, validation, testing = base.split()

	"""
	Setup experiment config
	"""

	#currently, only oversample
	# sampling_options = [Oversampling.Repeat]
	sampling_options = [Oversampling.Repeat, Oversampling.SmoteRegular]

	# learning_rule = stochastic gradient descent ('sgd'), 'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'
	learning_rule_options = ['momentum', 'sgd']
	#learning_rule_options = ['sgd', 'momentum','rmsprop']

	#following the SKNN docs
	#activation_function_options = ['Sigmoid']
	activation_function_options = ['Sigmoid', 'Rectifier', 'Tanh']

	#activation_function_options = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

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
	
	configDesc = {'opt_samp':'', 'opt_learning':'', 'activation_function_options':'', 'activation_function_options':'', 'topology_options':''}
	nConfig = 0
	
	#Create folder with timestamp
	mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	os.makedirs(mydir)

	config_results = []
	for opt_samp in sampling_options:

		if opt_samp != Oversampling.DontUse:
			configDesc['opt_samp'] = ''
			configDesc['opt_samp'] = opt_samp.name

			"""
			TRAINING OVER SAMPLE
			"""
			oversampler = Oversampler(opt_samp, training['data'], training['target'], True)
			training['data'], training['target'] = oversampler.balance()

			"""
			VALIDATION OVER SAMPLE
			"""
			oversampler = Oversampler(opt_samp,validation['data'], validation['target'],True )
			validation['data'], validation['target'] = oversampler.balance()

			"""
			DO NOT MAKE SENSE OVERSAMPLING OF TESTING SET
			"""
		base = {'training':training, 'validation': validation, 'testing': testing}

		for opt_learning in learning_rule_options:
			configDesc['opt_learning'] = ''
			configDesc['opt_learning'] = opt_learning
			for opt_top in topology_options:
				configDesc['topology_options'] = ''
				configDesc['topology_options'] = opt_top
				for opt_actvfunc in activation_function_options:
					configDesc['activation_function_options'] = ''
					configDesc['activation_function_options'] = opt_actvfunc
					configDir = os.path.join(mydir, 'config_' + str(nConfig))
					os.makedirs(configDir)
					config = Config(base, opt_learning, opt_top, opt_actvfunc, force_overfiting = False)

					#data storing for charts
					error_train = []
					error_valid = []

					def store_errors(avg_valid_error, avg_train_error, **_):
						error_train.append(avg_valid_error)
						error_valid.append(avg_train_error)

					'''LEARNING'''

					layers = [Layer(type=opt_actvfunc,name=topology['name'],units=topology['units']) for topology in opt_top];
					layers.append(Layer(type='Softmax',name="output_layer"))

					nn = Classifier(
					learning_rule = opt_learning,
				    layers=layers,
				    learning_rate=0.001,
				    n_iter=10,
				    valid_set=(base['validation']['data'],base['validation']['target']),
				    callback={'on_epoch_finish': store_errors},
				    verbose = verbose
				    )

					nn.fit(base['training']['data'],base['training']['target'])

					print('Testing')
					predictions = np.squeeze(np.asarray(nn.predict(base['testing']['data'])))
					target = base['testing']['target']

					errors = 0
					test_mse = 0
					for predicted, obj in zip(predictions,base['testing']['target']):
						predicted

						if predicted != obj:
							# print(' error')
							errors += 1
							test_mse += math.pow(predicted-obj, 2)
						
					test_mse = test_mse/float(len(predictions))

					"""
					PLOT AND CALCULATE METRICS
					"""

					#Confusion Matrix
					confusion_matrix = Metrics.plot_confusion_matrix(target, predictions, configDir)

					#MSE (Training and Validation)
					Metrics.plot_mse_curve(np.array(error_train), np.array(error_valid), configDir)

					#Area Under ROC Curve
					roc_area = Metrics.plot_roc_curve(target, predictions, configDir)

					#precision
					acurracy = ((len(base['testing']['data'])-errors)/len(base['testing']['data']))*100

					print("acurracy:", acurracy,'%')
					print('errors',errors,'of', len(base['testing']['data']))
					
					current_config_result = {'config':configDesc, 'results':{'mse':test_mse,'confusion':confusion_matrix,'roc':roc_area,'precision':acurracy}}
					config_results.append(current_config_result)

					Metrics.saveConfig(os.path.join(configDir, 'config-results.json'), current_config_result)

					nConfig = nConfig+1
		
	text = 'var configs = ['
	for config in config_results[:-1]:
		text += str(config) + ','
	
	text += str(config_results[-1]) + '];'
	Metrics.saveConfig('config.js', text)