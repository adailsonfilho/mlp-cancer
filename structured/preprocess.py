import numpy as np
from enums import Sampling

class Data:

	def __init__(self, path=None, separator=',', foreachline = None,verbose = False, target=None, data=None):

		self.path = path
		self.separator = separator
		self.all_data = []

		if verbose: print('>>> Preprocessing initialized')

		# A function for each line preprocess shuould be given, if note, consider a simple csv convertion		
		if(foreachline == None):

			if verbose: print('>>> No functino defined for preprocessing, using default csv.')

			def csvline_to_list(line):
				return [ float(val) for val in line.split(self.separator)]

			foreachline = csvline_to_list

		if verbose: print('>>> Loading file')

		with open(path, 'r') as csv:

		    for str_line in csv:
		    	line = foreachline(str_line)
		    	self.all_data.append(line)

		self.all_data = np.array(self.all_data)
		
		if target == None:
			self.target = self.all_data[:,-1].astype('int')
		else:
			self.target = target

		self.target_options= np.array(sorted(list(set(self.target))))
		
		if data == None:
			self.data = self.all_data[:,:-1].astype('int')
		else:
			self.data = data

		self.dimension = self.data.shape[1]

		if verbose:
			print('>>> Data lines total:',self.data.shape[0])
			print('>>> Targets Options Amount:', self.target_options.shape[0])
			print('>>> Dimension:',self.dimension)

	def split(self, training_percent = 0.5, validation_percent = 0.25, testing_percent = 0.25):

		assert training_percent + validation_percent + testing_percent == 1.0

		"""
		Split by classes
		"""	
		classes = []

		for opt in self.target_options:
			class_opt = data[:,opt == self.target]

			#randomize samples in each class
			np.random.shuffle(class_opt)
			classes.append(class_opt)

		training_sets = []
		validation_sets = []
		testing_sets = []

		#Preparando conjunto de treinamento - 50%
		for class_opt in classes:

			"""
			Training Data Set
			"""
			train_index = int(len(opt)*training_percent)
			training_sets.append(class_opt[:train_index])

			"""
			Validating
			"""
			validation_index = int(len(opt)*(training_percent+validation_percent))
			validation_sets.append(class_opt[train_index:validation_index])

			"""
			Testing
			"""
			testing_sets.append(class_opt[validation_index:])
		return training_sets, validation_sets, testing_sets