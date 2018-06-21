from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from opsy import *

import numpy as np
import csv
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random

class entity():
    def __init__(self, ent):
        self.id = int(ent[0])
        self.features = ent[1:-1].astype(int)
        self.label = int(ent[-1].split('_')[1])-1
        
class testEntity():
    def __init__(self, ent):
        self.id = int(ent[0])
        self.features = ent[1:].astype(int)
        
class secondEntity():
    def __init__(self, ent):
        self.id = int(ent[0])
        self.features = ent[1:].astype(float)
        

class Capsule(object):
    def __init__(self, sess, input_size=93, input_width=None, crop=True,
        batch_size=64,  c_dim=1, primary_dim=10, digit_dim=16, reg_para = 0.0005, n_conv=256,
        n_primary=32, n_digit=9, recon_h1=64, recon_h2=128, checkpoint_dir=None):

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size

        self.input_size = input_size

        self.c_dim = c_dim

        self.primary_dim = primary_dim
        self.digit_dim = digit_dim

        self.n_conv = n_conv
        self.n_primary=n_primary
        self.n_digit = n_digit

        self.recon_h1 = recon_h1
        self.recon_h2 = recon_h2
    
        self.recon_output = self.input_size

        self.reg_para = reg_para
        
        self.load_input()
        self.build_model()

    def build_model(self):
        self.primary_caps_layer = CapsConv(self.primary_dim, name='primary_caps')
        self.digit_caps_layer= CapsConv(self.digit_dim, name='digit_caps')

        self.input_x = tf.placeholder(dtype=tf.float32, shape=(None, 93), name='inputs')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=(self.batch_size), name='labels')
        self.is_training = tf.placeholder(tf.bool)
        self.recon_with_label = tf.placeholder_with_default(True, shape=(), name='reconstruction_with_label')

        self.first_fc = conv1d(self.input_x, 32, self.is_training)
        self.primary_caps = self.primary_caps_layer(self.first_fc, self.n_digit, is_training=self.is_training)
        self.digit_caps = self.digit_caps_layer(self.primary_caps, self.n_digit, is_training=self.is_training) ## shape: [batch_size, num_caps, dim_caps]
        
        with tf.variable_scope("prediction") as scope:
            self.logit = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=-1)) #[batch_size, num_caps]
            self.prob = tf.nn.softmax(self.logit)
            self.pred_label = tf.argmax(self.prob, axis=1)

        with tf.variable_scope("reconstruction") as scope:
            self.recon = self.reconstruction(name='reconstruction')

        with tf.variable_scope("loss") as scope:
            self.m_loss = margin_loss(self.logit, self.input_y)
            self.r_loss = reconstruction_loss(self.input_x, self.recon)
            self.loss = tf.add(self.m_loss, self.reg_para * self.r_loss)

            self.m_loss_sum = tf.summary.scalar("margin_loss", self.m_loss)
            self.r_loss_sum = tf.summary.scalar("reconstruction_loss", self.r_loss)
            self.loss_sum = tf.summary.scalar("total_loss", self.loss)
            
            self.acc = self.accuracy(self.input_y, self.prob)
            self.acc_sum = tf.summary.scalar("accuracy", self.acc)

        self.counter = tf.Variable(0, name='global_step', trainable=False)

    def reconstruction(self, name='reconstruction'):
        if self.sess.run(self.recon_with_label) == True:
            recon_mask = tf.one_hot(self.input_y, depth=self.n_digit, name='mask_output') #shape = [batch_size, 9]
        else:
            mask_target = self.pred_label  #shape = [batch_size]
            recon_mask = tf.one_hot(mask_target, depth=self.n_digit, name='mask_output') #shape = [batch_size, 9]

        recon_mask = tf.reshape(recon_mask, [-1, self.n_digit, 1], name='reshape_mask_output') # shape [batch_size, 9 ,1]

        recon_mask = tf.multiply(self.digit_caps, recon_mask, name='mask_result')
        recon_mask = tf.layers.flatten(recon_mask, name='mask_input')

        with tf.variable_scope(name) as scope:
            hidden1 = fc_layer(recon_mask, self.recon_h1, activation='relu',name='hidden1')
            hidden2 = fc_layer(hidden1, self.recon_h2, activation='relu',name='hidden2')
            output = fc_layer(hidden2, self.recon_output, activation='sigmoid',name='reconstruction')

        return output

    def train(self, restore=0):
        opt = layers.optimize_loss(loss=self.loss,
                                global_step=self.counter,
                                learning_rate=1e-3,
                                summaries = None,
                                optimizer = tf.train.AdamOptimizer,
                                clip_gradients = 0.1)

        if restore == 0:
            tf.global_variables_initializer().run()
        else:
            self.restore()

        self.summary_op = tf.summary.merge_all()

        batch_num = int(len(self.train_data)/self.batch_size)

        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

        for epoch in range(3):
            seed = 100
            np.random.seed(seed)
            np.random.shuffle(self.train_data)

            for idx in range(batch_num-1):
                start_time = time.time()
                
                batch = self.train_data[idx*self.batch_size: (idx+1)*self.batch_size]
                batch_x = [data.features for data in batch]
                batch_y = [data.label for data in batch]

                feed_dict = {self.input_x: batch_x, self.input_y: batch_y, self.is_training: True}

                _, loss, train_accuracy, summary_str =  self.sess.run([opt, self.loss, self.acc, self.summary_op], feed_dict=feed_dict) #add summary opt.
                total_count = tf.train.global_step(self.sess, self.counter)
                self.writer.add_summary(summary_str, total_count)

                if total_count % 100 == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, train_loss: %.8f, train_accurracy: %.8f" \
                    % (epoch, idx, batch_num-1, time.time() - start_time, loss, train_accuracy))

    def validation_check(self):
        val_num = int(len(self.test_data)/self.batch_size)
        val_loss, val_accuracy = 0.0, 0.0
        for idx in range(val_num-1):
            valid = self.test_data[idx*self.batch_size: (idx+1)*self.batch_size]
            valid_x = [data.features for data in valid]
            valid_y = [data.label for data in valid]
            
            feed_dict = {self.input_x: valid_x, self.input_y: valid_y, self.is_training: False}
            loss, accuracy = self.sess.run([self.loss, self.acc], feed_dict=feed_dict)
            val_loss += loss
            val_accuracy += accuracy
    
        val_loss /=(val_num-1)
        val_accuracy /= (val_num-1)
        print("[*] Validation: loss = %.8f, accuracy: %.8f"\
          %(val_loss, val_accuracy))

    def test_check(self):
        self.test_load_input()
        temp = 0
        predict = []
        f = open("C:\\Users\\Moon\\Desktop\\Moon's\\kaggle\\submitFile1.csv",'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['id','Class_1','Class_2','Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
    
        for batch_index in range(len(self.testData) // self.batch_size):
            batch = self.testData[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            batch_id = [data.id for data in batch]
            batch_data = [data.features for data in batch]
            
            feed_dict = {self.input_x: batch_data, self.is_training: False}
            test_out = self.sess.run(self.prob, feed_dict=feed_dict)
            test_out_list = test_out.tolist()
            for answer_id, answer in zip(batch_id, test_out_list):
                answer.insert(0, answer_id)
                predict.append(answer)
            
            temp = batch_index
    
        batch = self.testData[(temp + 1) * self.batch_size :]
        batch_id = [data.id for data in batch]
        batch_data = [data.features for data in batch]
        
        feed_dict = {self.input_x: batch_data, self.is_training: False}
        test_out = self.sess.run(self.prob, feed_dict=feed_dict)
        test_out_list = test_out.tolist()
        for answer_id, answer in zip(batch_id, test_out_list):
            answer.insert(0, answer_id)
            predict.append(answer)

        for submit in predict:
            writer.writerow(submit)
        f.close()
        
        print('done!!')

    def test_reconstruction(self):
        num_recon = self.batch_size
        num_test = len(self.test_data)
        sample_idx = list(np.random.choice(num_test, num_recon))
        rec = []
        rec2 = []
        
        test_x = [data.features for data in self.test_data]
        test_y = [data.label for data in self.test_data]
        test_id = [data.id for data in self.test_data]
        self.sample_x, self.sample_y, self.sample_id = test_x[0:100], test_y[0:100], test_id[0:100]
        
        f = open("C:\\Users\\Moon\\Desktop\\Moon's\\kaggle\\reconstruction_without_labels.csv",'w', newline='')
        writer = csv.writer(f)

        feed_dict = {self.input_x: self.sample_x, self.input_y: self.sample_y, self.is_training: False, self.recon_with_label: False}
        recon_images = self.sess.run(self.recon, feed_dict=feed_dict)
        
        recon_images_list = recon_images.tolist()
        for recon, rec_id in zip(recon_images_list, self.sample_id):
            recon.insert(0, rec_id)
            rec.append(recon)
        
        for submit in rec:
            writer.writerow(submit)
        
        f.close()
        
        print("[*] Reconstruction data of samples are saved without labels")
        
        f = open("C:\\Users\\Moon\\Desktop\\Moon's\\kaggle\\reconstruction_with_labels.csv",'w', newline='')
        writer = csv.writer(f)

        feed_dict = {self.input_x: self.sample_x, self.input_y: self.sample_y, self.is_training: False, self.recon_with_label: True}
        recon_images = self.sess.run(self.recon, feed_dict=feed_dict)
        
        recon_images_list = recon_images.tolist()
        for recon, rec_id in zip(recon_images_list, self.sample_id):
            recon.insert(0, rec_id)
            rec2.append(recon)
        
        for submit in rec2:
            writer.writerow(submit)
        
        f.close()
        
        print("[*] Reconstruction data of samples are saved with labels")

    def accuracy(self, y, y_pred):
        #y: true one-hot label
        #y_pred: predicted logit
        y = tf.one_hot(y, depth=9)
        correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy
    
    def load_input(self):
        with open("C:\\Users\\Moon\\Desktop\\Moon's\\kaggle\\train.csv", 'r') as f:
            reader = csv.reader(f)
            table_all = np.array(list(reader))
            
        data = [entity(ent) for ent in table_all[1:]]
        random.shuffle(data)
        
        self.train_data = data[0:int(len(data) * 0.8)]
        self.test_data = data[int(len(data) * 0.8):]
        
        print('data_setting_done')
        
    def test_load_input(self):
        with open("C:\\Users\\Moon\\Desktop\\Moon's\\kaggle\\test.csv", 'r') as f:
            reader = csv.reader(f)
            table_all = np.array(list(reader))
        
        self.testData = [testEntity(ent) for ent in table_all[1:]]
        
        print('test_data_setting_done')
        
    def save(self, checkpoint_path, name):
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        checkpoint_name_path =os.path.join(checkpoint_path,'%s.ckpt'% name)

        value_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
        saver=tf.train.Saver(value_list)
        saver.save(self.sess, checkpoint_name_path)
        
        print('save done!!')
        
    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "log\\Capsule.ckpt")
    
        print('restore done!!')