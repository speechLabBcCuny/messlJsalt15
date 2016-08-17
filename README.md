# Model-Based EM Source Separation and Localization Wrappers

This repo contains various wrappers and drivers for running MESSL
experiments for the 2015 Jelenik Speech and Language Technologies
workshop and for the OSU submission to CHiME3.

The actual MESSL code is in the [main MESSL repo](http://github.com/mim/messl).

## Examples of running this using SLURM on OSU's cluster

Step 1: Run MESSL
```matlab
enhance_wrapper(@(X,fail,fs,file) stubI_messlMc(X,fail,fs,file,1,0,0.12,1,'bestMic','mrfCompatExpSched',[0 0 0 0 0 0 0 0 .2],'mrfHardCompatExp',5,'mrfCompatFile','~mandelm/data8/messlData/ibmNeighborCountsSimple.mat', 'mrfLbpIter', 4), '/data/corpora/chime3/simulatedMoreMovement/isolated_MESSL', '~/data1/chime3/out/moreMovement/messlMcMvdrMrf.2Hard5Lbp4', [1 1], 0, 0, 1);
```

Step 2a: Replay MESSL, version 1: Mask-based noise estimate, MESSL IPD-based steering vector
```matlab
enhance_wrapper(@(X,fail,fs,file) stubI_replayMessl(X,fail,fs,file,'~/data1/chime3/out/messlMcMvdrMrf.2Hard5Lbp4Slate/data/','','souden','ipd','mask',1), '/data/corpora/chime3/CHiME3/data/audio/16kHz/isolated/', '~/data1/chime3/out/replaySlateXcMaskIpd', [1 1], 0, 1, 1);
```

Step 2b: Replay MESSL, version 2: Mask-based noise estimate, Souden-based "steering vector"
```matlab
enhance_wrapper(@(X,fail,fs,file) stubI_replayMessl(X,fail,fs,file,'~/data1/chime3/out/messlMcMvdrMrf.2Hard5Lbp4Slate/data/','','souden','','mask',1), '/data/corpora/chime3/CHiME3/data/audio/16kHz/isolated/', '~/data1/chime3/out/replaySlateXcMaskSouden', [1 1], 0, 1, 1);
```
