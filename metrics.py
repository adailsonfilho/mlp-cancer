import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

class Metrics:

	def plot_confusion_matrix(target, predictions, path, title='Confusion matrix', cmap=plt.cm.Blues):
		cm = confusion_matrix(target, predictions)
		np.set_printoptions(precision=2)
		plt.figure()
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(2)
		plt.xticks(tick_marks, '0', rotation=45)
		plt.yticks(tick_marks, '1')
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		Metrics.save(title, path)
		print('Confusion Matrix:')
		print(cm)
		
		return cm.tolist()
		# plt.show()

	def plot_roc_curve(y_test, y_score, path, title='ROC Curve'):
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		fpr, tpr, _ = roc_curve(y_test, y_score)
		roc_auc = auc(fpr, tpr)

	    # Plot all ROC curves
		plt.figure()
		print(roc_auc)
		plt.plot(fpr, tpr, label='ROC curve (area = ' + str(roc_auc) + ' )')
		#plt.grid()
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend(loc="lower right")
		Metrics.save(title, path)
		
		return roc_auc
	    # plt.show()

	def plot_mse_curve(error_train, error_valid, path, title='MSE Curve'):
	    plt.plot(error_train,label="MSE - Training")
	    plt.plot(error_valid,label="MSE - Validation")
	    plt.legend(loc="best")
	    Metrics.save(title, path)
	    # plt.show()


	def save(fname, path, ext='png', close=True, verbose=True):

		# Set the directory and filename
		#directory = 'results'
		filename = "%s.%s" % (fname, ext)

		# If the directory does not exist, create it
		#if not os.path.exists(directory):
		#    os.makedirs(directory)

		# The final path to save to
		savepath = os.path.join(path, filename)

		if verbose:
		    print("Saving figure to '%s'..." % savepath),

		# Actually save the figure
		plt.savefig(savepath)

		# Close it
		if close:
		    plt.close()

		if verbose:
		    print("Done")

	def saveConfig(path, config):
		with open(path, 'w') as output:
			output.write(str(config))
			