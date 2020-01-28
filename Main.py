import argparse
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def image_to_tensor(image_path, image_size) :
    img = image.load_img(image_path, target_size=image_size[:2])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with TF_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    return graph_def

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="DenseNet-121", type=str, help="Specifiy the name of the model")
    parser.add_argument("image_test", default="elephant", type=str, help="Specifiy the image for predicition")
    args = parser.parse_args()

    models_infos = { 
        "DenseNet-121" : {"path": "./Models/DenseNet-121","trt_graph" : "DenseNet-121-trt-graph.pb",
        "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
        "Inception-v3" : {"path" : "./Models/Inception-v3", "trt_graph" : "Inception-v3-trt-graph.pb",
        "output_names" : "predictions/Softmax", "input_names" : "input_1"},
        "MobileNet-v1" : {"path" : "./Models/MobileNet-v1", "trt_graph" : "MobileNet-v1-trt-graph.pb",
        "output_names" : "act_softmax/Softmax", "input_names" : "input_1"},
        "MobileNet-v2" :  {"path" : "./Models/MobileNet-v2", "trt_graph" : "MobileNet-v2-trt-graph.pb",
        "output_names" : "Logits/Softmax", "input_names" : "input_1"},
        "NASNet" : {"path" : "./Models/NASNet", "trt_graph" : "NASNet-trt-graph.pb",
        "output_names" : "predictions/Softmax", "input_names" : "input_1"},
        "ResNet-50" : {"path" : "./Models/ResNet-50", "trt_graph" : "ResNet-50-trt-graph.pb",
        "output_names" : "fc1000/Softmax", "input_names" : "input_1"},
        "Xception" : {"path" : "./Models/Xception", "trt_graph" : "Xception-trt-graph.pb",
        "output_names" : "predictions/Softmax", "input_names" : "input_2"}
        }
    
    model_info = models_infos[args.model_name]

    TF_graph = tf.Graph()
    trt_graph_path = model_info['path']+'/Saved-Model/'+model_info['trt_graph']
    trt_graph = get_frozen_graph(trt_graph_path)

    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    # Get graph input size
    for node in trt_graph.node:
        if 'input_' in node.name:
            size = node.attr['shape'].shape
            image_size = [size.dim[i].size for i in range(1, 4)]
            break

    # input and output tensor names.
    input_names = model_info['input_names']
    output_names = model_info['output_names']
    input_tensor_name = input_names + ":0"
    output_tensor_name = output_names + ":0"
    output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

    # Optional image to test model prediction.
    img_path = 'data/'+args.image_test+'.jpg'

    x = image_to_tensor(img_path, (224,224,3))

    feed_dict = {
        input_tensor_name: x
    }

    # in order to get rid of some unreletable values, cuz the first ones are always slow
    for i in range(0,5):
        one_prediction = tf_sess.run(output_tensor, feed_dict)
        
    time.sleep(3)
        
    # for a batch size = 1, to get a better results, we'll mesure the average time of many exeuction
    times = []
    with TF_graph.as_default():
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




