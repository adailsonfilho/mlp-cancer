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

def calc_confusion_matrix(vp,fp,fn,vn,pos_len, neg_len):
	print('>> MATRIX CONFUSION CHECK')
	cm = [[vp,fp],[fn,vn]]

	#VP
	cm [0][0] /= pos_len

	#FP
	cm [0][1] = 1 - cm [0][0]

	#VN
	cm [1][1] /= neg_len

	#FN
	cm [1][0] = 1 - cm [1][1]

	return cm


if __name__ == '__main__':

	verbose = True

	"""
	Pre-process
	"""

	base = Data('mammography-consolidated.csv', verbose=verbose,normalize=True)
	training, validation, testing = base.split()

	"""
	Setup experiment config
	"""

	#samplinh
	# sampling_options = [Oversampling.SmoteRegular]
	sampling_options = [Oversampling.DontUse, Oversampling.Repeat, Oversampling.SmoteRegular, Undersampling.ClusterCentroids, Undersampling.SMOTEENN]

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

		if opt_samp == Oversampling.DontUse:

			w_train = np.zeros(base['training']['data'].shape[0])
			w_train[base['training']['target'] == 0] = 0.2
			w_train[base['training']['target'] == 1] = 1.8

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
						    learning_rate=0.001,
						    n_iter=10000,
						    valid_set=(base['validation']['data'],base['validation']['target']),
						    callback={'on_epoch_finish': store_errors},
						    verbose = verbose
						    )

					if opt_samp == Oversampling.DontUse:
						nn.fit(base['training']['data'],base['training']['target'],w_train)
					else:
						nn.fit(base['training']['data'],base['training']['target'])

					print('Testing')
					predictions = np.squeeze(np.asarray(nn.predict(base['testing']['data'])))
					prob_predictions = nn.predict_proba(base['testing']['data'])
					target = base['testing']['target']
					targetByClass = np.array([0,0])
					
					errors_total = 0
					vp = 0
					fp = 0
					vn = 0
					fn = 0

					test_mse = 0
					for predicted, obj in zip(predictions,base['testing']['target']):
						predicted

						if predicted != obj:
							# print(' error')
							errors_total += 1
							test_mse += math.pow(predicted-obj, 2)

						if predicted == 1:
							if obj == 1:
								vp +=1
							elif obj == 0:
								fp +=1
						elif predicted == 0:
							if obj == 1:
								fn +=1
							elif obj == 0:
								vn +=1


						if obj == 0:
							targetByClass = np.vstack((targetByClass, [1, 0]))
						else:
							targetByClass = np.vstack((targetByClass, [0, 1]))

					#Remove first row
					targetByClass = np.delete(targetByClass, (0), axis=0)

					test_mse = test_mse/float(len(predictions))

					"""
					PLOT AND CALCULATE METRICS
					"""

					pos_len = len(base['testing']['data'][base['testing']['target']==1])
					neg_len = len(base['testing']['data'][base['testing']['target']==0])
					confusion_matrix_percentage = calc_confusion_matrix(vp,fp,fn,vn,pos_len,neg_len)

					#Confusion Matrix
					Metrics.plot_confusion_matrix(confusion_matrix_percentage, configDir)

					#MSE (Training and Validation)
					Metrics.plot_mse_curve(np.array(error_train), np.array(error_valid), configDir)

					#Area Under ROC Curve
					roc_area = Metrics.plot_roc_curve(targetByClass, prob_predictions, configDir)

					#precision
					acurracy = ((len(base['testing']['data'])-errors_total)/len(base['testing']['data']))*100

					print("acurracy:", acurracy,'%')
					print('errors',errors_total,'of', len(base['testing']['data']))
					
					configDesc = {'opt_samp':opt_samp.name, 'opt_learning':opt_learning, 'activation_function_options':opt_actvfunc, 'topology_options':opt_top}

					current_config_result = {'config':configDesc, 'results':{'mse':test_mse,'confusion':{'true_positive':confusion_matrix_percentage[0][0],'false_positive':confusion_matrix_percentage[0][1],'false_negative':confusion_matrix_percentage[1][0],'true_negative':confusion_matrix_percentage[1][1]},'roc':roc_area,'precision':acurracy}}
					config_results.append(current_config_result.copy())

					Metrics.saveConfig(os.path.join(configDir, 'config-results.json'), current_config_result)

					nConfig = nConfig+1
					current_config_result = {}


	
	text = 'var configs = ['
	for config in config_results[:-1]:
		text += str(config) + ','
	
	text += str(config_results[-1]) + '];'

	if (os.path.exists('config.js')):
		os.remove('config.js')
	
	Metrics.saveConfig('config.js', text)
	Metrics.copyDirectory(mydir, latest)