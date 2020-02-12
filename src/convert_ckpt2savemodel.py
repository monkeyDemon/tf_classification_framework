# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:47 2019

convert ckpt to savemodel format pb

使用Tensorflow Serving server来部署模型，必须选择SavedModel格式

在转换权重文件格式的同时，本脚本同时改造模型的输入节点使得模型支持base64的string输入
这样便于模型部署后的http请求

@author: as
"""
import os
import platform
import tensorflow as tf
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2

from utils.config_utils import load_config_file, mkdir_if_nonexist, import_model_by_networkname
from data import dataset_utils

flags = tf.app.flags
flags.DEFINE_string('action', None, 'command action: print or convert')
flags.DEFINE_string('config_path', '', 'path of the config file')
flags.DEFINE_integer('ckpt_idx', -1, 'the epoch idx of ckpt models')
FLAGS = flags.FLAGS



# old function
#def prepareImage(image, config_dict):
#
#    # 定义预处理方法
#    # 这样部署后，只需直接传入base64的string即可直接得到结果
#    img_decoded = tf.image.decode_png(image, channels=3)
#    #img_decoded = tf.image.decode_jpeg(image, channels=3)
#
#    # 与本demo一致的tensor版本预处理，保持长宽比将长边resize到224，然后padding到224*224
#    shape = tf.shape(img_decoded)
#    height = tf.to_float(shape[0])
#    width = tf.to_float(shape[1])
#    image_size = config_dict['DATASET']['IMAGE_SIZE']
#    scale = tf.cond(tf.greater(height, width),
#                    lambda: image_size / height,
#                    lambda: image_size / width)
#    new_height = tf.to_int32(tf.rint(height * scale))
#    new_width = tf.to_int32(tf.rint(width * scale))
#    resized_image = tf.image.resize_images(img_decoded, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
#    padd_image = tf.image.resize_image_with_crop_or_pad(resized_image, image_size, image_size)
#    padd_image = tf.cast(padd_image, tf.uint8)
#    return padd_image



def prepareImage(image, config_dict):
    # get preprocess info
    image_size = config_dict['DATASET']['IMAGE_SIZE']

    # get dataset mean std info
    output_paras = config_dict['OUTPUT']
    experiment_base_dir = os.path.join(output_paras['OUTPUT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
    model_save_dir = os.path.join(experiment_base_dir, 'weights')
    mean_std_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')
    dataset_rgb_mean, dataset_rgb_std = dataset_utils.load_dataset_mean_std_file(mean_std_file)
    r_mean, g_mean, b_mean = dataset_rgb_mean
    r_std, g_std, b_std = dataset_rgb_std

    # 定义预处理方法
    # 这样部署后，只需直接传入base64的string即可直接得到结果
    img_decoded = tf.image.decode_png(image, channels=3)
    #img_decoded = tf.image.decode_jpeg(image, channels=3)

    # 与本demo一致的tensor版本预处理，保持长宽比将长边resize到224，然后padding到224*224
    shape = tf.shape(img_decoded)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.cond(tf.greater(height, width),
                    lambda: image_size / height,
                    lambda: image_size / width)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    resized_image = tf.image.resize_images(img_decoded, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)

    # normalization
    R = tf.ones([new_height, new_width, 1], dtype=tf.float32) * r_mean
    G = tf.ones([new_height, new_width, 1], dtype=tf.float32) * g_mean
    B = tf.ones([new_height, new_width, 1], dtype=tf.float32) * b_mean
    rgb_img_mean = tf.concat([R,G,B], axis=2)
    img_centered = tf.subtract(resized_image, rgb_img_mean)
    img_normalize = tf.divide(img_centered, [r_std, g_std, b_std])
    #img_normalize = tf.cast(img_normalize, dtype=tf.float32) 
    
    padd_image = tf.image.resize_image_with_crop_or_pad(img_normalize, image_size, image_size)
    return padd_image



def saveModel(savemodel_save_dir, graph, sess):
    freezing_graph = graph
    images = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("images:0"))
    scores = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("score_list:0"))
    classes = tf.saved_model.utils.build_tensor_info(freezing_graph.get_tensor_by_name("classes:0")) 
    
    builder = tf.saved_model.builder.SavedModelBuilder(savemodel_save_dir)
    freezing_graph = graph
    builder.add_meta_graph_and_variables(
      sess,
      ['serve'], # tag
      signature_def_map={
          'serving_default': tf.saved_model.signature_def_utils.build_signature_def(
                      inputs = {'input':images},
                      outputs = {'scores':scores, 'classes':classes},
                      method_name = tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
                     )
      },
      clear_devices=True
    )
    builder.save()
    print("Exported SavedModel into %s" % savemodel_save_dir)



def freeze_graph(ckpt_path, savemodel_save_dir, Model, config_dict):
    '''
    :param ckpt_path:
    :param savemodel_save_dir: savemodel PB模型保存路径
    :return:
    '''

    graph = tf.Graph()
    with graph.as_default():
        train_layers = []

        images = tf.placeholder(tf.string, name="images")
        images_rank = tf.cond(tf.less(tf.rank(images), 1), lambda: tf.expand_dims(images, 0), lambda: images)
        orignal_inputs = tf.map_fn(fn=lambda inp: prepareImage(inp, config_dict), elems=images_rank, dtype=tf.float32)

        # init network
        model = Model(config_dict)
    
        # build model
        num_classes = config_dict['MODEL']['NUM_CLASSES']
        labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')
        model.build_model(orignal_inputs, labels, is_training=False)

        predictions = model.predict_op
        classes = tf.argmax(predictions, axis=1, name='classes')

        # restore ckpt model and convert to savemodel
        saver_all = tf.train.Saver(tf.all_variables())
        with tf.Session() as sess:
            saver_all.restore(sess, ckpt_path)
            saveModel(savemodel_save_dir, graph, sess)
 
 

def print_op_names(checkpoint_path):
    '''
    print all the operation names in the model
    
    convert ckpt to pb need you know the op name of input and output op
    so you can use this function to find out
    '''
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        sess = tf.Session(config = config)
        with sess.as_default():
            meta_path = checkpoint_path + '.meta'
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, checkpoint_path)
    
            op_list = sess.graph.get_operations()
            for op in op_list:
                print(op.name)
                print(op.values())


def _get_warmup_data_dir(dataset_root_dir, label_file):
    # use the first category in trainset as warmup data
    with open(label_file, 'r') as reader:
        for line in reader:
            items = line.rstrip().split(':')
            category_name = items[1]
            break
    warmup_data_dir = os.path.join(dataset_root_dir, category_name)
    return warmup_data_dir
    

def build_warmup_data(savemodel_dir, config_dict):

    serving_batch = config_dict['SERVING']['SERVE_BATCH']
    model_spec_name = config_dict['SERVING']['NAME']

    output_paras = config_dict['OUTPUT']
    experiment_base_dir = os.path.join(output_paras['OUTPUT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
    model_save_dir = os.path.join(experiment_base_dir, 'weights')

    label_file = os.path.join(model_save_dir, 'labels.txt')
    trainset_root_dir = config_dict['DATASET']['DATASET_ROOT_DIR']
    warmup_data_dir = _get_warmup_data_dir(trainset_root_dir, label_file)

    assets_extra_dir = os.path.join(savemodel_dir, 'assets.extra')
    mkdir_if_nonexist(assets_extra_dir, raise_error=False)
    warmup_save_path = os.path.join(assets_extra_dir, 'tf_serving_warmup_requests')

    # load warmup images
    cnt = 1
    image_str_list = []
    for img_name in os.listdir(warmup_data_dir):
        if cnt > serving_batch:
            break
        img_path = os.path.join(warmup_data_dir, img_name)
        image_str = (open(img_path,'rb').read())
        image_str_list.append(image_str)
        cnt += 1

    with tf.python_io.TFRecordWriter(warmup_save_path) as writer:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_spec_name
        request.model_spec.signature_name = "serving_default"
        #request.inputs['input'].ParseFromString(tf.make_tensor_proto(image_str, shape=[1]).SerializeToString())
        request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(image_str_list, dtype=tf.string))

        log = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())



def _get_ckpt_path(ckpt_save_dir, ckpt_idx):
    if ckpt_idx == -1:
        # use the lastest model
        ckpt_path = tf.train.latest_checkpoint(ckpt_save_dir) 
        return ckpt_path

    for file_name in os.listdir(ckpt_save_dir):
        if file_name.endswith('data-00000-of-00001'):
            epoch_idx = int(file_name.split('epoch')[1].split('.')[0])
            if ckpt_idx == epoch_idx:
                ckpt_name = file_name.split('.data-00000-of-00001')[0]
                ckpt_path = os.path.join(ckpt_save_dir, ckpt_name)
                return ckpt_path
    raise RuntimeError("Not found ckpt")
                
         

def main(_):
    config_path = FLAGS.config_path
    config_dict = load_config_file(config_path)
    ckpt_idx = FLAGS.ckpt_idx

    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict['GPU_OPTIONS']['GPU_DEVICES'])

    # output parameters
    output_paras = config_dict['OUTPUT']
    experiment_base_dir = os.path.join(output_paras['OUTPUT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
    ckpt_save_dir = os.path.join(experiment_base_dir, 'weights')
    
    ckpt_path = _get_ckpt_path(ckpt_save_dir, ckpt_idx)

    savemodel_save_dir = os.path.join(ckpt_save_dir, 'savemodel_base64')  
    mkdir_if_nonexist(savemodel_save_dir, raise_error=False)
    savemodel_save_dir = os.path.join(savemodel_save_dir, '1')
    mkdir_if_nonexist(savemodel_save_dir, raise_error=False)

    # import model by network_name, after that use can use Model
    network_name = config_dict['MODEL']['NETWORK_NAME']
    import_str = import_model_by_networkname(network_name)

    python_version = platform.python_version()
    if python_version.startswith('2.'):
        exec(import_str)  # python 2
    else:
        namespace = {}
        exec(import_str, namespace) 
        Model = namespace['Model']

    if FLAGS.action == 'print':
        print_op_names(ckpt_path)
    elif FLAGS.action == 'convert':
        freeze_graph(ckpt_path, savemodel_save_dir, Model, config_dict)

        build_warmup_data(savemodel_save_dir, config_dict)
    else:
        raise RuntimeError("parameter action error! use print or convert")


if __name__ == '__main__':
    tf.app.run()
