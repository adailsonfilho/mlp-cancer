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
	    # plt.show()

	def plot_mse_curve(X, y, path, title='MSE Curve'):
	    degrees = [1, 4, 15]
	    
	    true_fun = lambda X: np.cos(1.5 * np.pi * X)
	    y = true_fun(X) + y
	    
	    plt.figure(figsize=(14, 5))
	    for i in range(len(degrees)):
	        ax = plt.subplot(1, len(degrees), i + 1)
	        plt.setp(ax, xticks=(), yticks=())
	    
	        polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
	        linear_regression = LinearRegression()
	        pipeline = Pipeline([("polynomial_features", polynomial_features),
	                             ("linear_regression", linear_regression)])
	        pipeline.fit(X[:, np.newaxis], y)
	    
	        # Evaluate the models using crossvalidation
	        scores = cross_validation.cross_val_score(pipeline,
	            X[:, np.newaxis], y, scoring="mean_squared_error", cv=10)
	    
	        X_test = np.linspace(0, 1, 100)
	        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
	        plt.plot(X_test, true_fun(X_test), label="True function")
	        plt.scatter(X, y, label="Samples")
	        plt.xlabel("x")
	        plt.ylabel("y")
	        plt.xlim((0, 1))
	        plt.ylim((-2, 2))
	        plt.legend(loc="best")
	        plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
	            degrees[i], -scores.mean(), scores.std())
	        )
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
			