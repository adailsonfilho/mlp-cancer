import numpy as np

class Data:

	def __init__(self, path=None, separator=',', foreachline = None,verbose = False, target=None, data=None, normalize=False):

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
			self.target = self.all_data[:,-1]
		else:
			self.target = target

		self.target_options= np.array(sorted(list(set(self.target))))
		
		if data == None:
			self.data = self.all_data[:,:-1]
		else:
			self.data = data

		if normalize:
				min_ref = self.data.min()
				max_ref = self.data.max()

				if verbose:
					print('>> NORMALIZING: Min = ',min_ref,"; Max = ", max_ref)

				self.data -= min_ref
				self.data /= (max_ref-min_ref)

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

		training_sets = None
		training_targets = None

		validation_sets = None
		validation_targets = None

		testing_sets = None
		testing_targets = None

		for opt in self.target_options:
			class_opt = self.data[opt == self.target]

			#randomize samples in each class
			np.random.shuffle(class_opt)

			"""
			Training Data Set
			"""
			train_index = int(len(class_opt)*training_percent)
			training_set = class_opt[:train_index]

			if training_sets == None:
				training_sets = training_set
			else:
				training_sets = np.concatenate((training_sets,training_set))

			training_target = np.array([opt for i in range(len(training_set))])

			if training_targets == None:
				training_targets = training_target
			else:
				training_targets = np.concatenate((training_targets,training_target))

			"""
			Validating
			"""
			validation_index = int(len(class_opt)*(training_percent+validation_percent))			
			validation_set = class_opt[train_index:validation_index]

			if validation_sets == None:
				validation_sets = validation_set
			else:
				validation_sets = np.concatenate((validation_sets,validation_set))

			validation_target = np.array([opt for i in range(len(validation_set))])
			
			if validation_targets == None:
				validation_targets = validation_target
			else:
				validation_targets = np.concatenate((validation_targets,validation_target))



			"""
			Testing
			"""
			testing_set = class_opt[validation_index:]

			if testing_sets == None:
				testing_sets = testing_set
			else:
				testing_sets = np.concatenate((testing_sets,testing_set))

			testing_target = np.array([opt for i in range(len(testing_set))])
			
			if testing_targets == None:
				testing_targets = testing_target
			else:
				testing_targets = np.concatenate((testing_targets,testing_target))



		return {'data':training_sets, 'target': training_targets}, {'data':validation_sets,'target':validation_targets}, {'data':testing_sets,'target':testing_targets}