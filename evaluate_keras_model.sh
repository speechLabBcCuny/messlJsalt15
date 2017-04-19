#!/bin/bash

echo 'model path = ' $1
combination_option="$2"
model_dir=`dirname "$1"`
model_name=`basename "$1"`
echo "$model_dir"
echo "$model_name"
result_dir="$model_dir""/"${model_name::-5}"-results"
echo "$result_dir"
#given a path to keras model, evaluate the WER and PESQ on CHiME3
#mkdir "$result_dir"

#1 generate all the masks
#python generate_masks_from_model.py "$1"

#2 generate the audio files by applying the masks
lstm_mask_dir="$result_dir""/masks/"
audio_dir="$result_dir""/audios/"
matlab -r "job = batch(generate_audio_from_masks('$combination_option', '$lstm_mask_dir', '$audio_dir'),'Pool',10)"
# matlab generate_audio($combination_option, "$lstm_mask_dir", result_dir, audio_dir,2,5) &
# matlab generate_audio($combination_option, "$lstm_mask_dir", result_dir, audio_dir,3,5) &
# matlab generate_audio($combination_option, "$lstm_mask_dir", result_dir, audio_dir,4,5) &
# matlab generate_audio($combination_option, "$lstm_mask_dir", result_dir, audio_dir,5,5) &
#3 evaluate PESQ

#4 evaluate WER using kaldi
