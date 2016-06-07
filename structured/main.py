import numpy as np
from preprocess import Data
from config import Sampling, Config


if __name__ == '__main__':

	"""
	Pre-process
	"""

	base = Data('..\mammography-consolidated.csv',verbose=True)

	"""
	Setup experiment config
	"""

	#base, sampling, learning_algorithm, topology, activation_function, force_overfiting = False

	sampling_options = [Sampling.Smote, Sampling.Repeat, Sampling.Dontuse]

	learning_algorithm_options = []

	activation_function_options = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']

	topology_options = [
		[
			{name:'hidden', units:5}
		],
		[
			{name:'hidden', units:5},
			{name:'hidden', units:2},
		],
		[
			{name:'hidden', units:50},
			{name:'hidden', units:30},
		]

	]

