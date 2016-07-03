import numpy as np
from preprocess import Data
# from config import Config
from oversampler import Oversampler
from undersampler import Undersampler
from enums import Oversampling
from enums import Undersampling
from sknn.platform import cpu32, threading, threads4
from sknn.mlp import Layer, Classifier
import datetime
import os, sys
import math
import shutil

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

	#samplinh
	sampling_options = [Oversampling.DontUse, Oversampling.Repeat, Oversampling.SmoteRegular, Undersampling.ClusterCentroids]
	# sampling_options = [Undersampling.ClusterCentroids, Undersampling.SMOTEENN]

	# learning_rule = stochastic gradient descent ('sgd'), 'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'
	learning_rule_options = ['momentum']
	# learning_rule_options = ['momentum', 'sgd']
	#learning_rule_options = ['sgd', 'momentum','rmsprop']

	#following the SKNN docs
	activation_function_options = ['Sigmoid']
	#activation_function_options = ['Sigmoid', 'Rectifier', 'Tanh']

	#activation_function_options = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

	#based on W.I.W.T. - What I Want To
	topology_options = [
		# [
		# 	{'name':'hidden', 'units':5}
		# ],
		# [
		# 	{'name':'hidden', 'units':20}
		# ],
		# [
		# 	{'name':'hidden1', 'units':5},
		# 	{'name':'hidden2', 'units':2},
		# ],
		[
			{'name':'hidden1', 'units':5},
			{'name':'hidden2', 'units':4},
			{'name':'hidden3', 'units':3}
		]
		# [
		# 	{'name':'hidden1', 'units':50},
		# 	{'name':'hidden2', 'units':30},
		# ]
	]
	
	nConfig = 0
	
	#Create folder with timestamp
	mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	latest = os.path.join(os.getcwd(), 'latest')
	
	if (os.path.exists(latest)):
		shutil.rmtree(latest)
		
	os.makedirs(mydir)

	config_results = []
	configDir = ''
	for opt_samp in sampling_options:

		if opt_samp != Oversampling.DontUse:

			if opt_samp in Oversampling:

				"""
				TRAINING OVER SAMPLE
				"""
				oversampler = Oversampler(opt_samp, training['data'], training['target'], True)
				training['data'], training['target'] = oversampler.balance()

				"""
				VALIDATION OVER SAMPLE
				"""
				oversampler = Oversampler(opt_samp,validation['data'], validation['target'],True)
				validation['data'], validation['target'] = oversampler.balance()

			elif opt_samp in Undersampling:
				"""
				TRAINING OVER SAMPLE
				"""
				undersampler = Undersampler(opt_samp, training['data'], training['target'], True)
				training['data'], training['target'] = undersampler.balance()

				"""
				VALIDATION OVER SAMPLE
				"""
				undersampler = Undersampler(opt_samp,validation['data'], validation['target'],True)
				validation['data'], validation['target'] = undersampler.balance()
			else:
				raise('Nonexistent sampling type: '+opt_samp.name)

			"""
			DOES NOT MAKE SENSE OVERSAMPLING OF TESTING SET, THINK ABOUT IT...
			"""
		base = {'training':training, 'validation': validation, 'testing': testing}

		for opt_learning in learning_rule_options:
			for opt_top in topology_options:
				for opt_actvfunc in activation_function_options:

					configDir = os.path.join(mydir, 'config_' + str(nConfig))
					os.makedirs(configDir)

					# config = Config(base, opt_learning, opt_top, opt_actvfunc, force_overfiting = False)

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
				    learning_rate=0.0001,
				    n_iter=10000,
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
					confusion_matrix_percentage = np.round(np.multiply(np.divide(confusion_matrix, np.array([target.size])),np.array([100])),2).tolist()

					#MSE (Training and Validation)
					Metrics.plot_mse_curve(np.array(error_train), np.array(error_valid), configDir)

					#Area Under ROC Curve
					roc_area = Metrics.plot_roc_curve(target, predictions, configDir)

					#precision
					acurracy = ((len(base['testing']['data'])-errors)/len(base['testing']['data']))*100

					print("acurracy:", acurracy,'%')
					print('errors',errors,'of', len(base['testing']['data']))
					
					configDesc = {'opt_samp':opt_samp.name, 'opt_learning':opt_learning, 'activation_function_options':opt_actvfunc, 'topology_options':opt_top}

					current_config_result = {'config':configDesc, 'results':{'mse':test_mse,'confusion':confusion_matrix_percentage,'roc':roc_area,'precision':acurracy}}
					config_results.append(current_config_result.copy())

					Metrics.saveConfig(os.path.join(configDir, 'config-results.json'), current_config_result)

					nConfig = nConfig+1
		
	text = 'var configs = ['
	for config in config_results[:-1]:
		text += str(config) + ','
	
	text += str(config_results[-1]) + '];'
	
	if (os.path.exists('config.js')):
		os.remove('config.js')
	
	Metrics.saveConfig('config.js', text)
	Metrics.copyDirectory(mydir, latest)