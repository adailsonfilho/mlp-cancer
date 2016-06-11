from unbalanced_dataset.over_sampling import SMOTE
from enums import Oversampling
import numpy as np
# import ipdb

class Repeater:

	def __init__(self,verbose):
		self.verbose = verbose

	
	def fit_transform(self, data, target):

		target_options= np.array(sorted(list(set(target))))

		#split in diffent targets values
		classes = []

		max_size = 0

		if self.verbose: print('Spliting by targets')
		for opt in target_options:
			class_opt = data[opt == target]

			#randomize samples in each class
			np.random.shuffle(class_opt)
			classes.append(class_opt)

			if len(class_opt) > max_size: max_size = len(class_opt)

		if self.verbose: print('Majoritary class with:',max_size,'samples')

		over_data = []
		over_target = []

		if self.verbose: print('Starting oversampling')
		for i,class_opt in enumerate(classes):
			extra = []
			while(len(class_opt) + len(extra) < max_size):
				extra.append(class_opt[np.random.randint(len(class_opt))])

			if(extra != []):
				over_class_opt = np.concatenate((class_opt,np.array(extra)))
			else:
				over_class_opt = class_opt

			if over_data == []:
				over_data = over_class_opt
			else: 
				over_data = np.concatenate((over_data,over_class_opt))
			
			if over_target == []:
				over_target = np.array([i for j in range(max_size)])
			else:
				over_target = np.concatenate((over_target,np.array([i for j in range(max_size)])))

		if self.verbose: print('Oversampling finished')

		return over_data, over_target

class Oversampler:

	def __init__(self,kind,data,target,verbose = False, ratio = 'auto'):

		assert len(data) == len(target)
		self.data = data
		self.target = target

		if kind in [Oversampling.SmoteRegular, Oversampling.SmoteSVM]:
			if verbose: print('> SMOTE')
			#adpter
			if Oversampling.SmoteRegular: smotekind = 'regular'
			elif Oversampling.SmoteSVM: smotekind = 'svm'

			# Oversampling por SMOTE
			self.oversampler = SMOTE(kind=smotekind, verbose = verbose, ratio=ratio)
		elif kind in [ Oversampling.Repeat]:
			if verbose: print('> REPEATER')
			self.oversampler = Repeater(verbose = verbose)

	def balance(self):
		return self.oversampler.fit_transform(self.data, self.target)