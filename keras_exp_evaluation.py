### Scripts that runs a full evaluation of a trained model

# imports
from __future__ import print_function
import sys
import os
import messlkeras as mk

# # set verbosity
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# allow growth
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
config.log_device_placement=True
set_session(tf.Session(config=config))

# tf.logging.set_verbosity(tf.logging.FATAL)


# preliminaries
# script arguments
# name of this script
script_name = sys.argv[0]

#directory with the trained model
model_dir = sys.argv[1]

# desired GPU
gpu_num =  sys.argv[2]

# if want to use CPU
if gpu_num=='-1':
	gpu_num=''

#choose a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
print("Running script on GPU", gpu_num)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num # desired GPU


print("Evaluation of model in {}".format(model_dir))

# 1.- run the model on the dt05 and et05 sets, creating the predicted masks for those
#
print("Extracting all the predicted masks")
mk.predict_masks_from_model(model_dir, model_type='best')

# 4. wer_scorer_for_chime3.sh