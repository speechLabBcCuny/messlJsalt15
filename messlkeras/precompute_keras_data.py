# pre compute keras data (all but PSA targets)
# needed to save RAM

from __future__ import print_function
import numpy as np
import time

import cPickle as pickle

import sys
sys.path.append('/home/felix/Research/messlJsalt15/')
import messlkeras as mk


print("Time: ", time.strftime('%Y-%m-%d_%T'))

# prepare all the lists
# tr_lists, dt_lists, et_lists = mk.prep_chime3_lists()

precomp_dir = "/scratch/felix/MESSLKERAS/pre-computed-data/"
tr_lists = pickle.load( open( precomp_dir + '/tr_lists.pkl', "rb" ) )
dt_lists = pickle.load( open( precomp_dir + '/dt_lists.pkl', "rb" ) )
et_lists = pickle.load( open( precomp_dir + '/et_lists.pkl', "rb" ) )

[chime_spects_noisy_tr, messl_soft_masks_tr, iaf_masks_tr, psf_masks_tr, messl_spects_mvdr_tr] = tr_lists
[chime_spects_noisy_dt, messl_soft_masks_dt, iaf_masks_dt, psf_masks_dt, messl_spects_mvdr_dt] = dt_lists
# [messl_spects_noisy_et, messl_soft_masks_et, iaf_masks_et, psf_masks_et, messl_spects_mvdr_et] = et_lists # not used

print("Number of training and testing files")
print(len(chime_spects_noisy_tr))
print(len(messl_soft_masks_tr))
print(len(iaf_masks_tr))
print(len(psf_masks_tr))
print(len(messl_spects_mvdr_tr))
print()
print(len(chime_spects_noisy_dt))
print(len(messl_soft_masks_dt)) 
print(len(iaf_masks_dt))
print(len(psf_masks_dt))
print(len(messl_spects_mvdr_dt))

# check that they match

#check that the filenames all match
if ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in messl_soft_masks_tr]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in iaf_masks_tr]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in psf_masks_tr]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in messl_spects_mvdr_tr]) :
    print("Number of training files:", len(chime_spects_noisy_tr))
else:
    raise Exception("Training filenames for inputs and targets do not match! Exiting")
    
#check that the filenames all match
if ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in messl_soft_masks_dt]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in iaf_masks_dt]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in psf_masks_dt]) \
    and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in messl_spects_mvdr_dt]) :
    print("Number of validation files:", len(chime_spects_noisy_dt))
else:
    raise Exception("Validation filenames for inputs and targets do not match! Exiting")


save_dir = precomp_dir


# # make smaller for debugging
# size = 2
# chime_spects_noisy_tr = chime_spects_noisy_tr[0:size]
# messl_soft_masks_tr = messl_soft_masks_tr[0:size]
# iaf_masks_tr = iaf_masks_tr[0:size]
# psf_masks_tr = psf_masks_tr[0:size]
# messl_spects_mvdr_tr = messl_spects_mvdr_tr[0:size]

# chime_spects_noisy_dt = chime_spects_noisy_dt[0:size]
# messl_soft_masks_dt = messl_soft_masks_dt[0:size]
# iaf_masks_dt = iaf_masks_dt[0:size]
# psf_masks_dt = psf_masks_dt[0:size]
# messl_spects_mvdr_dt = messl_spects_mvdr_dt[0:size]
# save_dir = precomp_dir+"/dev/"


# prepare the rest
# tr
input_shape=(-1, 150, 513)
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_chime_spects_noisy_tr,_ = mk.prep_data_for_keras(chime_spects_noisy_tr, data_type="input_spect", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_chime_spects_noisy_tr: {} bytes".format(keras_chime_spects_noisy_tr.nbytes))
print(keras_chime_spects_noisy_tr.dtype)
print("keras_chime_spects_noisy_tr: {} shape".format(keras_chime_spects_noisy_tr.shape))
np.save(save_dir + '/keras_chime_spects_noisy_tr.npy', keras_chime_spects_noisy_tr)
del(keras_chime_spects_noisy_tr) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_messl_soft_masks_tr,_ = mk.prep_data_for_keras(messl_soft_masks_tr, data_type="input_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_messl_soft_masks_tr: {} bytes".format(keras_messl_soft_masks_tr.nbytes))
print(keras_messl_soft_masks_tr.dtype)
print("keras_messl_soft_masks_tr: {} shape".format(keras_messl_soft_masks_tr.shape))
np.save(save_dir + '/keras_messl_soft_masks_tr.npy', keras_messl_soft_masks_tr)
del(keras_messl_soft_masks_tr) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_iaf_masks_tr,_ = mk.prep_data_for_keras(iaf_masks_tr, data_type="target_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_iaf_masks_tr: {} bytes".format(keras_iaf_masks_tr.nbytes))
print(keras_iaf_masks_tr.dtype)
print("keras_iaf_masks_tr: {} shape".format(keras_iaf_masks_tr.shape))
np.save(save_dir + '/keras_iaf_masks_tr.npy', keras_iaf_masks_tr)
del(keras_iaf_masks_tr) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_psf_masks_tr,_ = mk.prep_data_for_keras(psf_masks_tr, data_type="target_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_psf_masks_tr: {} bytes".format(keras_psf_masks_tr.nbytes))
print(keras_psf_masks_tr.dtype)
print("keras_psf_masks_tr: {} shape".format(keras_psf_masks_tr.shape))
np.save(save_dir + '/keras_psf_masks_tr.npy', keras_psf_masks_tr)
del(keras_psf_masks_tr) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_messl_spects_mvdr_tr,_ = mk.prep_data_for_keras(messl_spects_mvdr_tr, data_type="target_spect", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_messl_spects_mvdr_tr: {} bytes".format(keras_messl_spects_mvdr_tr.nbytes))
print(keras_messl_spects_mvdr_tr.dtype)
print("keras_messl_spects_mvdr_tr: {} shape".format(keras_messl_spects_mvdr_tr.shape))
np.save(save_dir + '/keras_messl_spects_mvdr_tr.npy', keras_messl_spects_mvdr_tr)
del(keras_messl_spects_mvdr_tr) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

print()

# dt
input_shape=(-1, 150, 513)
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_chime_spects_noisy_dt,_ = mk.prep_data_for_keras(chime_spects_noisy_dt, data_type="input_spect", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_chime_spects_noisy_dt: {} bytes".format(keras_chime_spects_noisy_dt.nbytes))
print(keras_chime_spects_noisy_dt.dtype)
print("keras_chime_spects_noisy_dt: {} shape".format(keras_chime_spects_noisy_dt.shape))
np.save(save_dir + '/keras_chime_spects_noisy_dt.npy', keras_chime_spects_noisy_dt)
del(keras_chime_spects_noisy_dt) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_messl_soft_masks_dt,_ = mk.prep_data_for_keras(messl_soft_masks_dt, data_type="input_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_messl_soft_masks_dt: {} bytes".format(keras_messl_soft_masks_dt.nbytes))
print(keras_messl_soft_masks_dt.dtype)
print("keras_messl_soft_masks_dt: {} shape".format(keras_messl_soft_masks_dt.shape))
np.save(save_dir + '/keras_messl_soft_masks_dt.npy', keras_messl_soft_masks_dt)
del(keras_messl_soft_masks_dt) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_iaf_masks_dt,_ = mk.prep_data_for_keras(iaf_masks_dt, data_type="target_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_iaf_masks_dt: {} bytes".format(keras_iaf_masks_dt.nbytes))
print(keras_iaf_masks_dt.dtype)
print("keras_iaf_masks_dt: {} shape".format(keras_iaf_masks_dt.shape))
np.save(save_dir + '/keras_iaf_masks_dt.npy', keras_iaf_masks_dt)
del(keras_iaf_masks_dt) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_psf_masks_dt,_ = mk.prep_data_for_keras(psf_masks_dt, data_type="target_mask", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_psf_masks_dt: {} bytes".format(keras_psf_masks_dt.nbytes))
print(keras_psf_masks_dt.dtype)
print("keras_psf_masks_dt: {} shape".format(keras_psf_masks_dt.shape))
np.save(save_dir + '/keras_psf_masks_dt.npy', keras_psf_masks_dt)
del(keras_psf_masks_dt) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))

keras_messl_spects_mvdr_dt,_ = mk.prep_data_for_keras(messl_spects_mvdr_dt, data_type="target_spect", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("keras_messl_spects_mvdr_dt: {} bytes".format(keras_messl_spects_mvdr_dt.nbytes))
print(keras_messl_spects_mvdr_dt.dtype)
print("keras_messl_spects_mvdr_dt: {} shape".format(keras_messl_spects_mvdr_dt.shape))
np.save(save_dir + '/keras_messl_spects_mvdr_dt.npy', keras_messl_spects_mvdr_dt)
del(keras_messl_spects_mvdr_dt) # hopefully this works to save space
print("Time: ", time.strftime('%Y-%m-%d_%T'))


# now that the simple ones are done, we can do PSA targets
# tr
print("Time: ", time.strftime('%Y-%m-%d_%T'))
# prep angles
theta_chime_spects_noisy_tr, _ = mk.prep_data_for_keras(chime_spects_noisy_tr, data_type="theta", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("theta_chime_spects_noisy_tr: {} bytes".format(theta_chime_spects_noisy_tr.nbytes))
print(theta_chime_spects_noisy_tr.dtype)
print("theta_chime_spects_noisy_tr: {} shape".format(theta_chime_spects_noisy_tr.shape))

theta_keras_messl_spects_mvdr_tr, _ = mk.prep_data_for_keras(messl_spects_mvdr_tr, data_type="theta", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("theta_keras_messl_spects_mvdr_tr: {} bytes".format(theta_keras_messl_spects_mvdr_tr.nbytes))
print(theta_keras_messl_spects_mvdr_tr.dtype)
print("theta_keras_messl_spects_mvdr_tr: {} shape".format(theta_keras_messl_spects_mvdr_tr.shape))

print("computing theta")
theta = theta_keras_messl_spects_mvdr_tr - theta_chime_spects_noisy_tr
print("Time: ", time.strftime('%Y-%m-%d_%T'))

# save RAM
del(theta_keras_messl_spects_mvdr_tr)
del(theta_chime_spects_noisy_tr)

# to save
print("Time: ", time.strftime('%Y-%m-%d_%T'))
abs_keras_messl_spects_mvdr_tr,_ = mk.prep_data_for_keras(messl_spects_mvdr_tr, data_type="abs", input_shape=input_shape, time_limit=10800, verbose=True)

print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("applying cos(theta)")
abs_keras_messl_spects_mvdr_tr *= np.cos(theta)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
np.save(save_dir + '/keras_psa_targets_tr.npy', abs_keras_messl_spects_mvdr_tr)
# save RAM
del(abs_keras_messl_spects_mvdr_tr)
del(theta)


# dt
print("Time: ", time.strftime('%Y-%m-%d_%T'))
# prep angles
theta_chime_spects_noisy_dt, _ = mk.prep_data_for_keras(chime_spects_noisy_dt, data_type="theta", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("theta_chime_spects_noisy_dt: {} bytes".format(theta_chime_spects_noisy_dt.nbytes))
print(theta_chime_spects_noisy_dt.dtype)
print("theta_chime_spects_noisy_dt: {} shape".format(theta_chime_spects_noisy_dt.shape))

theta_keras_messl_spects_mvdr_dt, _ = mk.prep_data_for_keras(messl_spects_mvdr_dt, data_type="theta", input_shape=input_shape, time_limit=10800, verbose=True)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("theta_keras_messl_spects_mvdr_dt: {} bytes".format(theta_keras_messl_spects_mvdr_dt.nbytes))
print(theta_keras_messl_spects_mvdr_dt.dtype)
print("theta_keras_messl_spects_mvdr_dt: {} shape".format(theta_keras_messl_spects_mvdr_dt.shape))

print("computing theta")
theta = theta_keras_messl_spects_mvdr_dt - theta_chime_spects_noisy_dt
print("Time: ", time.strftime('%Y-%m-%d_%T'))

# save RAM
del(theta_keras_messl_spects_mvdr_dt)
del(theta_chime_spects_noisy_dt)

# to save
print("Time: ", time.strftime('%Y-%m-%d_%T'))
abs_keras_messl_spects_mvdr_dt,_ = mk.prep_data_for_keras(messl_spects_mvdr_dt, data_type="abs", input_shape=input_shape, time_limit=10800, verbose=True)

print("Time: ", time.strftime('%Y-%m-%d_%T'))
print("applying cos(theta)")
abs_keras_messl_spects_mvdr_dt *= np.cos(theta)
print("Time: ", time.strftime('%Y-%m-%d_%T'))
np.save(save_dir + '/keras_psa_targets_dt.npy', abs_keras_messl_spects_mvdr_dt)
# save RAM
del(abs_keras_messl_spects_mvdr_dt)
del(theta)
