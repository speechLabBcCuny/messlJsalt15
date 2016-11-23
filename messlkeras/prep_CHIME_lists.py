import os
import sys

def prep_CHIME_lists(noisy_spects_dir, clean_spect_dir masks_dir, set_type='train'):
    # given directories, creates lists of all the files (with paths) that will be used by keras
    # (once fed to  pred_data_xyz)
    # this function ensure that the filenames match to ensure proper order when fitting model
    # set_type = {'train', 'dev', 'test'}
    # this function is meant to be used on CHiME data only
    # noisy_spects_dir point to the directory in which the sub-directories contain matlab .mat files of noisy spectrograms
    # clean_spect_dir points to the cleaned spectrograms
    # (masks_dir points to the masks)
    # Note: this function does not check if noisy_spects_dir contain actual spectrograms, just '.mat' files with reasonable path
    
    if set_type not in ['train', 'dev', 'test']:
        # throw error if set_type is anything else
        raise Exception("set_type not in {'train', 'dev', 'test'}")
    else:
        # associate the right marker 
        set_marker = {'train':'tr05', 'dev':'dt05', 'test':'et05'}[set_type]
        
    # build file lists
    noisy_spects_list = [path+'/'+file \
                    for path,_,files in sorted(os.walk(noisy_spects_dir)) \
                    for file in sorted(files) \
                    if file.endswith('.mat') and set_marker in path and 'spects-clean' in path]

    clean_spects_list = [path+'/'+file \
                    for path,_,files in sorted(os.walk(clean_spects_dir)) \
                    for file in sorted(files) \
                    if file.endswith('.mat') and set_marker in path and 'spects-clean' in path]            
    
    masks_list = [path+'/'+file \
                    for path,_,files in sorted(os.walk(masks_dir)) \
                    for file in sorted(files) \
                    if file.endswith('.mat') and set_marker in path and 'masks' in path]
    
    # check that they are the same
    if [x.split('/')[-1] for x in masks_list] != [x.split('/')[-1] for x in spects_list]:
        raise Exception("Filenames do not match!")
    
    # return lists
    return noisy_spects_list, masks_list
    