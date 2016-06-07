import numpy as np

class Data:

	def __init__(self, path, separator=',', foreachline = None,verbose = False):

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
		
		self.targets = self.all_data[:,-1].astype('int')
		self.target_options= np.array(sorted(list(set(self.targets))))
		self.data = self.all_data[:,:-1].astype('int')
		self.dimension = self.data.shape[1]

		if verbose:
			print('>>> Data lines total:',self.data.shape[0])
			print('>>> Targets Options Amount:', self.target_options.shape[0])
			print('>>> Dimension:',self.dimension)

		

		