import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import os, sys, cv2, pdb
import argparse
#from networks.factory import get_network
from faster_rcnn_tf import faster_rcnn_tf

rnn = tf.nn.rnn
rnn_cell = tf.nn.rnn_cell
from spatial_transformer import transformer, batch_transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

Dims = 512

class ant_1dim_loc(object):
	def __init__(self, dataset, batch_size, n_times, faster_rcnn_model, memory_slots_num = 4, keep_prob = 0.9, start_time = 0, device = [0,1,2]):
		self.variables = []
		self.batch_size = batch_size
		self.start_time = start_time
		self.n_times = n_times
		self.keep_prob = keep_prob
		#self.faster_rcnn = get_network('VGGnet_test')
		#self.faster_rcnn = faster_rcnn_tf(faster_rcnn_model, 1)
		self.faster_rcnn = None
		self.model_variable()
		self.dataset = dataset
		self.device = device
		self.memory_slots_num = memory_slots_num

	def model_variable(self):

		# environment input soft-attention for environment stream
		self.W_a_f1 = weight_variable([4096, Dims])
		self.b_a_f1 = bias_variable([Dims])
		self.variables.append(self.W_a_f1)
		self.variables.append(self.b_a_f1)

		self.W_a_f2 = weight_variable([Dims, Dims])
		self.b_a_f2 = bias_variable([Dims])
		self.variables.append(self.W_a_f2)
		self.variables.append(self.b_a_f2)

		self.W_p_f1 = weight_variable([2, Dims])
		self.b_p_f1 = bias_variable([Dims])
		self.variables.append(self.W_p_f1)
		self.variables.append(self.b_p_f1)

		self.W_p_f2 = weight_variable([Dims, Dims])
		self.b_p_f2 = bias_variable([Dims])
		self.variables.append(self.W_p_f2)
		self.variables.append(self.b_p_f2)

		self.W_f2 = weight_variable([2*Dims, Dims])
		self.b_f2 = bias_variable([Dims])
		self.variables.append(self.W_f2)
		self.variables.append(self.b_f2)

		self.W_f3 = weight_variable([Dims, 4096])
		self.b_f3 = bias_variable([4096])
		self.variables.append(self.W_f3)
		self.variables.append(self.b_f3)

		# %% Create a fully-connected layer:
		"""
		self.W_fc1 = weight_variable([Dims * 2, Dims])
		self.b_fc1 = bias_variable([Dims])
		self.variables.append(self.W_fc1)
		self.variables.append(self.b_fc1)
		
		self.W_fc2 = weight_variable([Dims, Dims])
		self.b_fc2 = bias_variable([Dims])
		self.variables.append(self.W_fc2)
		self.variables.append(self.b_fc2)	

		self.W_fc3 = weight_variable([Dims, 2])
		self.b_fc3 = bias_variable([2])
		self.variables.append(self.W_fc3)
		self.variables.append(self.b_fc3)	
		"""

	def model(self, sess):
		# %% Graph representation of our network

		x = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 30, 4096])
		boxes = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 30, 4])
		x_a = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 2, 4096])
		boxes_a = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 2, 4])
		boxes_y = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 30 + 1])
		y = tf.placeholder(tf.int32, [None, self.n_times-self.start_time])

		cross_entropy = []
		pred = []
		y_risky = []
		for time in xrange(0, self.n_times-self.start_time):
			if time > 0:
                		tf.get_variable_scope().reuse_variables()
		
			agent_x = x_a[:,time,0,:]
			agent_b = boxes_a[:,time,0,:]
			risky_x = x_a[:,time,1,:]
			risky_b = boxes_a[:,time,1,:]

			agent_c = tf.expand_dims(tf.transpose(tf.pack([(agent_b[:,0]+agent_b[:,2])/2,(agent_b[:,1]+agent_b[:,3])/2]),[1,0]),1)
			risky_c = tf.expand_dims(tf.transpose(tf.pack([(risky_b[:,0]+risky_b[:,2])/2,(risky_b[:,1]+risky_b[:,3])/2]),[1,0]),1)
			rpn_c = tf.concat(1, [tf.transpose(tf.pack([(boxes[:,time,:,0]+boxes[:,time,:,2])/2,(boxes[:,time,:,1]+boxes[:,time,:,3])/2]),[1,2,0]), risky_c])
			#risky_dis = tf.squared_difference(rpn_c[:,:,0] - risky_c[:,:,0], -(rpn_c[:,:,1] - risky_c[:,:,1]))
			risky_y = boxes_y[:,time,:]
			risky_x = tf.concat(1,[x[:,time,:,:],tf.expand_dims(risky_x,1)])
			ar_dis = tf.transpose(tf.pack([agent_c[:,:,0] - rpn_c[:,:,0], agent_c[:,:,1] - rpn_c[:,:,1]]),[1,2,0])

			agent_h1 = tf.nn.relu(tf.matmul(agent_x,self.W_a_f1)+self.b_a_f1)
			agent_h2 = tf.tile(tf.expand_dims(tf.nn.relu(tf.matmul(agent_h1,self.W_a_f2)+self.b_a_f2),1),[1,31,1])
			risky_h1 = tf.nn.relu(tf.batch_matmul(ar_dis, tf.tile(tf.expand_dims(self.W_p_f1,0),[self.batch_size,1,1]))+self.b_p_f1)
			risky_h2 = tf.nn.relu(tf.batch_matmul(risky_h1, tf.tile(tf.expand_dims(self.W_p_f2,0),[self.batch_size,1,1]))+self.b_p_f2)
			h1 = tf.concat(2,[agent_h2,risky_h2])
			h2 = tf.nn.relu(tf.batch_matmul(h1, tf.tile(tf.expand_dims(self.W_f2,0),[self.batch_size,1,1]))+self.b_f2)
			h3 = tf.batch_matmul(h2,tf.tile(tf.expand_dims(self.W_f3,0),[self.batch_size,1,1]))+self.b_f3
			scores = tf.sigmoid(tf.matrix_diag_part(tf.batch_matmul(h3,tf.transpose(risky_x,[0,2,1]))))
			cross_entropy.append(tf.cast(y[:,time],tf.float32) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(scores,risky_y)))

			"""
			for batch in xrange(self.batch_size):
				indices = tf.cast(tf.expand_dims(tf.range(0,31, 1), 1), tf.int32)
				concated = tf.concat(1, [indices, tf.expand_dims(risky_y[batch,:],1)])
				y_target = tf.sparse_to_dense(concated, tf.pack([31, self.dataset.n_classes]), 1.0, 0.0)
				cross_entropy.append(tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(scores[batch,:,:], y_target)))
			"""
			"""
			with tf.device('/gpu:'+str(self.device[3])):
				# %% We'll now reshape so we can connect to a fully-connected layer:
				h_fc1 = tf.nn.relu(tf.matmul(tf.concat(1,[ha, tf.reduce_mean(hr,1)]), self.W_fc1) + self.b_fc1)
				h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

				h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
				h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

				y_logits = tf.matmul(h_fc2_drop, self.W_fc3) + self.b_fc3

				indices = tf.cast(tf.expand_dims(tf.range(0,self.batch_size, 1), 1), tf.int32)
				concated = tf.concat(1, [indices, tf.expand_dims(y[:,time],1)])
				y_target = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.dataset.n_classes]), 1.0, 0.0)
				cross_entropy.append(tf.reduce_mean(
					tf.nn.softmax_cross_entropy_with_logits(y_logits, y_target)))
			"""
			pred.append(scores)
			y_risky.append(risky_y)

		cross_entropy = tf.reduce_mean(tf.pack(cross_entropy))
		# %% Monitor accuracy
		pred_ = tf.to_int32(tf.greater_equal(tf.pack(pred),0.5))
		correct_prediction = tf.equal(pred_, tf.cast(tf.pack(y_risky),tf.int32))
		accuracy = tf.cast(correct_prediction, 'float')

		return x, boxes, x_a, boxes_a, boxes_y, y, cross_entropy, accuracy, tf.pack(pred)
