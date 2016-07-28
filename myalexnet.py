################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
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

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#Caffe version weights 
#net_data = load("bvlc_alexnet.npy").item()
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
