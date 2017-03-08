import re
import os

def prep_list_for_keras(data_dir, reg_exp, verbose=False):

    ### prepare the list of .mat files containing the data, which will then be fed to prep_data_for_keras
    # data_dir is the file directory containing the data (which could be in subdirectories)
    # reg_exp is the regular expression that identifies the directories containing the data to use:
    # eg: for noisy spectrogram dev set use "messl-spects-noisy.*dt" 
    # so only .mat in directories with that pattern will beadded to the list
	# returns a list
    
    res_list = [path+'/'+file \
                    for path,_,files in sorted(os.walk(data_dir)) \
                    for file in sorted(files) \
                    if file.endswith('.mat') and bool(re.search(reg_exp,path)) ]
    
    if verbose: print "file_list in paths matching {} prepared. Make sure to check filenames for input<->target match!".format(reg_exp)
    return res_list
