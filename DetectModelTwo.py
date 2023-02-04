import os 

CUSTOM_MODEL_NAME = 'my_ssd_mobnet_3p5'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'CUSTOM_MODELS': os.path.join('CustomModels'),
    'CHECKPOINT_PATH': os.path.join('CustomModels', CUSTOM_MODEL_NAME), 
 }

files = {
    'PIPELINE_CONFIG':os.path.join(paths['CUSTOM_MODELS'], CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['CUSTOM_MODELS'], LABEL_MAP_NAME)
}

labels = [{'name':'car', 'id':1}, {'name':'minion', 'id':2}, {'name':'rack', 'id':3}, {'name':'mixer', 'id':4}, {'name':'sandwich', 'id':5}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-2')).expect_partial()

@tf.function
def detect_fn_roundTwo(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])