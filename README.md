# TensorFlowCode
Collection of code that I found useful for playing around with tensorFlow

First experiment: Getting to know graphs 
dumpGraph.py for loading a graph from a pb file and save as checkpoints to use tensorboard to visualize graph

Idea: found code that converts the standard AlexNet CNN from caffe to TensorFlow
Implement code to save this model as protobuf file and use it later from the TensorFlow-style pb-file

Conversion code used from this github repo 
https://github.com/guerzh/tf_weights

Two phases: 
1) saving the AlexNet model with constant weights in a protobuf file
alexNetSave.py
2) loading the model from the protobuf file and use it for inference
alexNetLoad.py

Code to convert variables to constants for saving the graph:
https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/tensorflow/python/framework/graph_util_impl.py#L178
