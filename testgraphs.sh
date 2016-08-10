#!/bin/bash
#Shell script to test the speed and output of 
#32 and 8 bit graphs

echo "Original classify_image_graph_def.pb"
STARTTIME=$(date +%s)
./bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/tmp/classify_image_graph_def.pb \
--labels=./imagenet_comp_graph_label_strings.txt 
ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task...\n\n"


echo "Quantized as 8 bits"
STARTTIME=$(date +%s)
#command block that takes time to complete...
#........
./bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/tmp/quantized_graph.pb \
--labels=./imagenet_comp_graph_label_strings.txt \
--input_width=299 \
--input_height=299 \
--input_mean=128 \
--input_std=128 \
--input_layer="Mul:0" \
--output_layer="softmax:0"
ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task...\n\n"
