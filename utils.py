#!/usr/bin/python
#coding: utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
	print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(2)
	plt.xticks(tick_marks, '0', rotation=45)
	plt.yticks(tick_marks, '1')
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	save(title)
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
