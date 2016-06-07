class Sampling:

    Kmeans, Smote, Repeat, DontUse = range(4)


class Config:

	def __init__(self, base, sampling = Sampling.DontUse, learning_algorithm, topology, activation_function, force_overfiting = False):
		self.base = base
		self.sampling = sampling
		self.learning_algorithm = learning_algorithm
		self.topology = topology
		self.activation_function = activation_function
		self.force_overfiting = force_overfiting


	"""
	Base
	"""

	"""
	Sampling
	"""

	"""
	Learning Algorithm
	"""

	"""
	Topology
	"""

	"""
	Activation Function
	"""

	"""
	Force overfiting?
	"""

