#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

output_names = ['fc1000/Softmax']
input_names = ['input_1']


# In[10]:


classification_graph = tf.Graph()
def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with classification_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_file, "rb") as f:
            #graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    return graph_def


# In[11]:

trt_graph = get_frozen_graph('.//Saved-Model/DenseNet-121-trt-graph.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# In[12]:

# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break

# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
print('input_names :',input_names,', tensor_name :',input_tensor_name)
output_tensor_name = output_names[0] + ":0"
print('input_names :',output_names,', tensor_name :',output_tensor_name)

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


# In[13]:


# Optional image to test model prediction.
img_path = '../../data/elephant.jpg'

img = image.load_img(img_path, target_size=image_size[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feed_dict = {
    input_tensor_name: x
}


# In[14]:


import time
# in order to get rid of some unreletable values, cuz the first ones are always slow
for i in range(0,5):
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    
time.sleep(3)
    
# for a batch size = 1, to get a better results, we'll mesure the average time of many exeuction
times = []
with classification_graph.as_default():
    with tf_sess :
        for i in range(50) :
            start_time = time.time()
            one_prediction = tf_sess.run(output_tensor, feed_dict)
            end_time = time.time()
            delta = (end_time - start_time)
            times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))


# In[ ]:




