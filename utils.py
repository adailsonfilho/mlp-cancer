#!/usr/bin/python
#coding: utf-8
import os
import matplotlib.pyplot as plt

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
