################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#with changes by Isabel Schwende
#
#Original code here: https://github.com/guerzh/tf_weights
#
################################################################################

from numpy import *
import os
import os.path
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf
from tensorflow.python.platform import gfile

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

LOG_DIR = '/tmp/alex_log'#adapt 
if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

################################################################################
#Read Image

x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
i = x_dummy.copy()
i[0,:,:,:] = (imread("poodle.png")[:,:,:3]).astype(float32)
i = i-mean(i)

x = tf.placeholder(tf.float32, (None,)+xdim)
################################################################################

# AlexNet structure explanation
#         conv(11, 11, 96, 4, 4, padding='VALID', name='Conv2D:0')
# 	  other operations -> Conv2D_X, BiasAdd, Reshape, Relu, LRN, MaxPool,XW_plus_b, init
#         softmax(name='Softmax:0'))

# Caffe version weights from previously saved protobuf file
# has to adapt the path here
model_file ='/home/isabeltf/tensorflow/modelZoo/test_data/graph.pb'
with gfile.FastGFile(model_file,'rb') as f:
	with tf.Graph().as_default() as import_graph:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def,name='')

		#init = tf.initialize_all_variables()
		with tf.Session() as sess:	
	
			#sess.run(init)
			result = sess.graph.get_tensor_by_name('Softmax:0')
			var16 = sess.graph.get_tensor_by_name('Variable_1:0')
			allops = tf.Graph.get_operations(sess.graph)
			var_list =tf.all_variables()
			
			#print var_list 
			#sess.run(tf.initialize_variables(var16))
			print result
			print var16
		
			t = time.time()
			output = sess.run(result,feed_dict ={'Variable:0':i})
		
			print time.time()-t
			#Output:
			#inds = argsort(output)[0,:]
			#for i in range(5):
			#	print class_names[inds[-1-i]], output[0, inds[-1-i]]
