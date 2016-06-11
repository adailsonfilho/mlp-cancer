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

	def plot_confusion_matrix(target, predictions, title='Confusion matrix', cmap=plt.cm.Blues):
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
		Metrics.save(title)
		print('Confusion Matrix:')
		print(cm)
		plt.show()

	def plot_roc_curve(y_test, y_score, title='ROC Curve'):
	    fpr = dict()
	    tpr = dict()
	    roc_auc = dict()
	    for i in range(0, 1):
	        #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
	        roc_auc[i] = auc(fpr[i], tpr[i])
	    
	    # Compute micro-average ROC curve and ROC area
	    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	    
	    #Plot ROC curves for the multiclass problem
	    # Compute macro-average ROC curve and ROC area
	    # First aggregate all false positive rates
	    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0, 1)]))
	    
	    # Then interpolate all ROC curves at this points
	    mean_tpr = np.zeros_like(all_fpr)
	    for i in range(0, 1):
	        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	    
	    # Finally average it and compute AUC
	    mean_tpr /= range(0, 1)
	    
	    fpr["macro"] = all_fpr
	    tpr["macro"] = mean_tpr
	    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
	    
	    # Plot all ROC curves
	    plt.figure()
	    plt.plot(fpr["micro"], tpr["micro"],
	             label='micro-average ROC curve (area = {0:0.2f})'
	                   ''.format(roc_auc["micro"]),
	             linewidth=2)
	    
	    plt.plot(fpr["macro"], tpr["macro"],
	             label='macro-average ROC curve (area = {0:0.2f})'
	                   ''.format(roc_auc["macro"]),
	             linewidth=2)
	    
	    for i in range(0, 1):
	        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	                                       ''.format(i, roc_auc[i]))
	    
	    plt.plot([0, 1], [0, 1], 'k--')
	    plt.xlim([0.0, 1.0])
	    plt.ylim([0.0, 1.05])
	    plt.xlabel('False Positive Rate')
	    plt.ylabel('True Positive Rate')
	    plt.title('Some extension of Receiver operating characteristic to multi-class')
	    plt.legend(loc="lower right")
	    Metrics.save(title)
	    plt.show()

	def plot_mse_curve(X, y, title='MSE Curve'):
	    degrees = [1, 4, 15]
	    
	    true_fun = lambda X: np.cos(1.5 * np.pi * X)
	    y = true_fun(X) + y
	    
	    plt.figure(figsize=(14, 5))
	    for i in range(len(degrees)):
	        ax = plt.subplot(1, len(degrees), i + 1)
	        plt.setp(ax, xticks=(), yticks=())
	    
	        polynomial_features = PolynomialFeatures(degree=degrees[i],
	                                                 include_bias=False)
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
	    Metrics.save(title)
	    plt.show()

	def save(fname, ext='png', close=True, verbose=True):

		# Set the directory and filename
		directory = 'results'
		filename = "%s.%s" % (fname, ext)

		# If the directory does not exist, create it
		if not os.path.exists(directory):
		    os.makedirs(directory)

		# The final path to save to
		savepath = os.path.join(directory, filename)

		if verbose:
		    print("Saving figure to '%s'..." % savepath),

		# Actually save the figure
		plt.savefig(savepath)

		# Close it
		if close:
		    plt.close()

		if verbose:
		    print("Done")