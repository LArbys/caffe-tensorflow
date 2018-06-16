import os,sys

import numpy as np
import tensorflow as tf

from ssnet_tf_code_plane0 import UResNet

data_p0 = np.load("ssnet_tf_data_plane0.npy")

data_node = tf.placeholder(tf.float32,shape=(1, 512, 512, 1))

unet = UResNet({'data': data_node},trainable=False)
print "Net defined"

blank = np.zeros( (1,512,512,1), dtype=np.float32 )

with tf.Session() as sess:

    # load the model
    unet.load( "ssnet_tf_data_plane0.npy", sess )

    # define the folder to export the SavedModel
    export_dir = "export"

    pred = sess.run(unet.get_output(), feed_dict={data_node:blank})


