from unbalanced_dataset.over_sampling import SMOTE
from enums import Oversampling
import numpy as np

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

		over_data = np.array([])
		over_target = np.array([])

		if self.verbose: print('Starting oversampling')
		for i,class_opt in enumerate(classes):
			extra = []
			while(len(class_opt) + len(extra) < max_size):
				extra.append(class_opt[np.random.randint(len(class_opt))])

			class_opt = np.concatenate((class_opt,np.array(extra)))
			over_data = np.concatenate((over_data,class_opt))
			over_target = np.concatenate((over_target,np.array([i for j in range(max_size)])))

		if self.verbose: print('Oversampling finished')

		return over_data, over_target

class Oversampler:

	def __init__(self,kind,data,target,verbose = False):

		assert len(data) == len(target)
		self.data = data
		self.target = target

		if kind in [ Oversampling.SmoteRegular, Oversampling.SmoteSVM]:
			# Oversampling por SMOTE
			self.oversampler = SMOTE(kind=kind, verbose = verbose)
		elif kind in [ Oversampling.Repeat]:
			self.oversampler = Repeater(verbose = verbose)

	def balance(self):
		return self.oversampler.fit_transform(self.data, self.target)