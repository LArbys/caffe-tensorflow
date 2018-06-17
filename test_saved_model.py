import os,sys,time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

tload = time.time()
predict_fn = predictor.from_saved_model("SaveModelSSNet")
tload = time.time()-tload
print "Time to load predictor: ",tload,"secs"

niters = 10
batch_size = 8

trun = time.time()
for i in range(niters):
    blank = np.random.rand(batch_size,512,512,1).astype(np.float32)
    pred = predict_fn( {"uplane":blank })
    diff = np.sum( pred["pred"]-blank )
    print "iter ",i,pred["pred"].shape,"diff=",diff

trun = time.time()-trun
print "Time for %d iterations: "%(niters),trun,"secs"
print "Time for one batch (size=%d) prediction: "%(batch_size),trun/float(niters),"secs/batch  ",trun/float(niters*batch_size),"secs/img"
