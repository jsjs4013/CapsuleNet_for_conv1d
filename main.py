import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import random

import os
import numpy as np

from model import Capsule

import tensorflow as tf

def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        CapsuleNet = Capsule(sess, input_size=93, batch_size = 100)
    
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
#         CapsuleNet.train_conv1d()
#         CapsuleNet.validation_check_test()
#         CapsuleNet.test_check_test()
        
#         CapsuleNet.train_test()
#         CapsuleNet.test_check_test()

        CapsuleNet.train(0)
        CapsuleNet.validation_check()
        CapsuleNet.test_check()
        CapsuleNet.save('log', 'Capsule')
        CapsuleNet.test_reconstruction()
        
if __name__ == '__main__':
    main(_)