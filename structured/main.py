import numpy as np
from preprocess import Data
from config import Config
from oversampler import Oversampler
from enums import Sampling


if __name__ == '__main__':

	"""
	Pre-process
	"""

	base = Data('..\mammography-consolidated.csv',verbose=True)
	training, validation, testing = base.split()

	"""
	Setup experiment config
	"""

	#currently, only oversample
	sampling_options = [Sampling.Smote, Sampling.Repeat, Sampling.DontUse]

	# learning_rule = stochastic gradient descent ('sgd'), 'momentum', 'nesterov', 'adadelta', 'adagrad', 'rmsprop'
	learning_rule_options = ['sgd', 'momentum','rmsprop']

	#following the SKNN docs
	activation_function_options = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

	#based on W.I.W.T. - What I Want To
	topology_options = [
		[
			{name:'hidden', units:5}
		],
		[
			{name:'hidden', units:20}
		],
		[
			{name:'hidden', units:5},
			{name:'hidden', units:2},
		],
		[
			{name:'hidden', units:5},
			{name:'hidden', units:4},
			{name:'hidden', units:3}
		],
		[
			{name:'hidden', units:50},
			{name:'hidden', units:30},
		]

	]

	for opt_samp in sampling_options:

		oversampler = Oversampler(opt_samp)
		data, target = oversampler.balance()

		for opt_learning in learning_rule_options:
			for opt_top in topology_options:
				for opt_actvfunc in activation_function_options:

					config = Config( base, learning_algorithm, topology, activation_function, force_overfiting = False)
