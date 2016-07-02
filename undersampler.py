from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN
from enums import Undersampling
import numpy as np

class Undersampler:

	def __init__(self,kind,data,target,verbose = False, ratio = 'auto'):

		assert len(data) == len(target)
		self.data = data
		self.target = target

		if kind in [Undersampling.ClusterCentroids]:
			if verbose: print('> CLUSTER CENTROIDS')

			# Undersampling por Cluster Centroids
			self.undersampler = ClusterCentroids(verbose = verbose, ratio=ratio)
		elif kind in [Undersampling.SMOTEENN]:
			if verbose: print('> SMOTEENN')

			# Undersampling por SMOTEENN
			self.undersampler = SMOTEENN(verbose = verbose, ratio=ratio)

	def balance(self):
		#return self.undersampler.fit_transform(self.data, self.target)
		return self.undersampler.fit_sample(self.data, self.target)