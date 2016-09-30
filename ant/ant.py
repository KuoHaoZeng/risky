import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import os, sys, cv2, pdb
import argparse
from networks.factory import get_network

rnn = tf.nn.rnn
rnn_cell = tf.nn.rnn_cell
from spatial_transformer import transformer, batch_transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

class ant(object):
	def __init__(self, dataset, batch_size, n_times, keep_prob = 0.9, start_time = 0, device = [0,1,2]):
		self.variables = []
		self.batch_size = batch_size
		self.start_time = start_time
		self.n_times = n_times
		self.keep_prob = keep_prob
		self.faster_rcnn = get_network('VGGnet_test')
		self.model_variable()
		self.dataset = dataset
		self.device = device

	def model_variable(self):
		# %% We'll setup the two-layer localisation network to figure out the
		# %% parameters for an affine transformation of the input
		# %% Create variables for fully connected layer
		"""
		self.W_fc_loc1 = weight_variable([7*7*512, 7*7])
		self.b_fc_loc1 = bias_variable([7*7])
		self.varables.append(self.W_fc_loc1)
		self.varables.append(self.b_fc_loc1)
		"""
		with tf.variable_scope("LSTM1") as vs:
			#self.lstm1 = rnn_cell.LSTMCell(7*7,7*7*512)
			#self.lstm1 = rnn_cell.BasicRNNCell(7*7,7*7*512)
			self.lstm1 = rnn_cell.GRUCell(7*7,7*7*512)
			self.variables += [v for v in tf.all_variables() if v.name.startswith(vs.name)]
		self.state1 = tf.zeros([self.batch_size, self.lstm1.state_size])

		self.W_fc_loc2 = weight_variable([7*7, 6])
		# Use identity transformation as starting point
		initial = np.array([[1., 0, 0], [0, 1., 0]])
		initial = initial.astype('float32')
		initial = initial.flatten()
		self.b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
		self.variables.append(self.W_fc_loc2)
		self.variables.append(self.b_fc_loc2)

		# %% We'll setup the first convolutional layer
		# Weight matrix is [height x width x input_channels x output_channels]
		filter_size = 3
		n_filters_1 = 512
		self.W_conv1 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_1])
		# %% Bias is [output_channels]
		self.b_conv1 = bias_variable([n_filters_1])
		self.variables.append(self.W_conv1)
		self.variables.append(self.b_conv1)

		with tf.variable_scope("LSTM2") as vs:
			#self.lstm2 = rnn_cell.LSTMCell(7*7,7*7*512)
			#self.lstm2 = rnn_cell.BasicRNNCell(7*7,7*7*512)
			self.lstm2 = rnn_cell.GRUCell(7*7,7*7*512)
			self.variables += [v for v in tf.all_variables() if v.name.startswith(vs.name)]
		self.state2 = tf.zeros([self.batch_size, self.lstm2.state_size])

		self.W_fc_loc4 = weight_variable([7*7, 6])
		# Use identity transformation as starting point
		initial = np.array([[1., 0, 0], [0, 1., 0]])
		initial = initial.astype('float32')
		initial = initial.flatten()
		self.b_fc_loc4 = tf.Variable(initial_value=initial, name='b_fc_loc4')
		self.variables.append(self.W_fc_loc4)
		self.variables.append(self.b_fc_loc4)

		# %% And just like the first layer, add additional layers to create
		# a deep net
		n_filters_2 = 512
		self.W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
		self.b_conv2 = bias_variable([n_filters_2])
		self.variables.append(self.W_conv2)
		self.variables.append(self.b_conv2)

		# %% Create a fully-connected layer:
		n_fc = 4096
		self.W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
		self.b_fc1 = bias_variable([n_fc])
		self.variables.append(self.W_fc1)
		self.variables.append(self.b_fc1)

		# %% And finally our softmax layer:
		with tf.variable_scope("LSTM3") as vs:
			#self.lstm3 = rnn_cell.LSTMCell(500,n_fc)
			#self.lstm3 = rnn_cell.BasicRNNCell(500,n_fc)
			self.lstm3 = rnn_cell.GRUCell(500,n_fc)
			self.variables += [v for v in tf.all_variables() if v.name.startswith(vs.name)]
		self.state3 = tf.zeros([self.batch_size, self.lstm3.state_size])
		
		self.W_fc2 = weight_variable([500, 2])
		self.b_fc2 = bias_variable([2])
		self.variables.append(self.W_fc2)
		self.variables.append(self.b_fc2)	

	def body(self, foreground_x, x, dets, i, inds):
		det = tf.gather(dets, tf.cast(inds,tf.int32))
		det = tf.cast(tf.gather(det,i), tf.int32)
		x1 = det[0]
		y1 = det[1]
		x2 = det[2]
		y2 = det[3]
		x_tmp = tf.image.crop_to_bounding_box(x,x1,y1,x2-x1,y2-y1)
		foreground_x = tf.maximum(foreground_x, tf.image.pad_to_bounding_box(x_tmp,360-y1,640-x1,360,640))
		i+=1
		return foreground_x, x, dets, i, inds

	def condition(self, foreground_x, x, dets, i, inds):
		return i < tf.shape(inds)[0]

	def get_foreground(self,foreground_x, x_org, dets_, inds):
		[foreground_x, _, _, _, _] = tf.while_loop(self.condition, self.body, [foreground_x, x_org, dets_, tf.constant(0), inds])
		return foreground_x

	def do_nothing(self, foreground_x):
		return foreground_x

	def model(self, sess):
		# %% Graph representation of our network

		x = tf.placeholder(tf.uint8, [None, self.n_times-self.start_time, 360, 640, 3])
		x = tf.cast(x,tf.float32)
		y = tf.placeholder(tf.int32, [None, self.n_times-self.start_time])

		cross_entropy = []
		for time in xrange(0, self.n_times-self.start_time):
			if time > 0:
                		tf.get_variable_scope().reuse_variables()
			batch_pool_5_1 = []
			batch_pool_5_2 = []
			obj_x = []
			
			for batch in xrange(self.batch_size):
				#[scores, boxes, pool_5] = im_detect(sess, self.faster_rcnn, x[batch,time,:,:,:])
				with tf.device('/gpu:'+str(self.device[0])):
					[scores, boxes, pool_5] = im_detect(sess, self.faster_rcnn, x[batch,time,:,:,:])
				num_boxes = tf.shape(pool_5)[0]
				batch_pool_5_1.append(tf.gather(pool_5,num_boxes-1))
				batch_pool_5_2.append(tf.gather(pool_5,num_boxes-1))
				"""
				cls_boxes = boxes[:, 4*7:4*(7 + 1)] # 7 is car
				cls_scores = scores[:, 7]
				#dets = tf.cast(tf.concat(1,[cls_boxes,tf.expand_dims(cls_scores,0)]),tf.float32)
				#keep = nms(dets, NMS_THRESH)
				#cls_boxes_nms = cls_boxes/tf.constant([640.,360.,640.,360.])
				keep = tf.image.non_max_suppression(cls_boxes,cls_scores,10,iou_threshold=0.3)
        			#dets_ = dets[keep, :]
				dets_ = tf.gather(cls_boxes, keep)
				dets_scores = tf.gather(cls_scores, keep)
				cls_boxes = boxes[:, 4*15:4*(15 + 1)] # 15 is person
				cls_scores = scores[:, 15]
				#dets = tf.cast(tf.concat(1,[cls_boxes,tf.expand_dims(cls_scores,0)]),tf.float32)
				#keep = nms(dets, NMS_THRESH)
				#cls_boxes_nms = cls_boxes/tf.constant([640.,360.,640.,360.])
				keep = tf.image.non_max_suppression(cls_boxes,cls_scores,10,iou_threshold=0.3)
        			#dets_ += dets[keep, :]
				#dets_ = tf.concat(0,[dets_, tf.gather(dets, keep)])
				dets_ = tf.concat(0, [dets_, tf.gather(cls_boxes, keep)])
				dets_scores = tf.concat(0, [dets_scores, tf.gather(cls_scores, keep)])
				inds = tf.where(tf.greater_equal(dets_scores, 0.5))[:,0]

				foreground_x = tf.zeros((tf.shape(x)[2:]))
				x_org = tf.gather(tf.gather(x,batch),time)
				[foreground_x, _, _, _, _] = tf.while_loop(self.condition, self.body, [foreground_x, x_org, dets_, tf.constant(0), inds])
				#foreground_x = tf.cond(tf.shape(inds)[0] > 0, self.get_foreground(foreground_x,x_org,dets_,inds), self.do_nothing(foreground_x))
				#for i in inds:
				#	x_tmp = tf.image.crop_to_bounding_box(x[batch,time,:,:,:],dets[i,1],dets[i,0],dets[i,3]-dets[i,1],dets[i,2]-dets[i,0])
				#	foreground_x += tf.image.pad_to_bounding_box(x_tmp,360-dets[i,1],640-dets[i,0],360,640)
				background_x = x[batch,time,:,:,:] - foreground_x
				with tf.device('/gpu:'+str(self.device[0])):
					[_,_,batch_pool_5_1_tmp] = im_detect(sess, self.faster_rcnn, background_x)
				num_boxes = tf.shape(batch_pool_5_1_tmp)[0]
				batch_pool_5_1.append(tf.gather(batch_pool_5_1_tmp,num_boxes-1))
				#idx = tf.gather(dets,tf.argmax(dets[:,-1],0))
				#x_tmp = tf.image.crop_to_bounding_box(x[batch,time,:,:,:],idx[1],idx[0],idx[3]-idx[1],idx[2]-idx[0])
				#foreground_x = tf.image.pad_to_bounding_box(x_tmp,360-idx[1],640-idx[0],360,640)
				obj_x.append(foreground_x)
				with tf.device('/gpu:'+str(self.device[0])):
					[_,_,batch_pool_5_2_tmp] = im_detect(sess, self.faster_rcnn, foreground_x)
				num_boxes = tf.shape(batch_pool_5_2_tmp)[0]
				batch_pool_5_2.append(tf.gather(batch_pool_5_2_tmp,num_boxes-1))
				"""
			
			obj_x = tf.pack(obj_x)
			batch_pool_5_1 = tf.pack(batch_pool_5_1)
			batch_pool_5_2 = tf.pack(batch_pool_5_2)
			x_vector = tf.reshape(batch_pool_5_1, [-1, 7*7*512])

			# %% Define the two layer localisation network
			#h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
			with tf.variable_scope("LSTM1"):
				[h_fc_loc1, self.state1] = self.lstm1( x_vector, self.state1 )

			# %% We can add dropout for regularizing and to reduce overfitting like so:
			h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, self.keep_prob)
			# %% Second layer
			h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, self.W_fc_loc2) + self.b_fc_loc2)

			# %% We'll create a spatial transformer module to identify discriminative
			# %% patches
			out_size = (7, 7)
			h_trans = transformer(batch_pool_5_2, h_fc_loc2, out_size)

			# %% Now we can build a graph which does the first layer of convolution:
			# we define our stride as batch x height x width x channels
			# instead of pooling, we use strides of 2 and more layers
			# with smaller filters.
			h_conv1 = tf.nn.relu(
			    tf.nn.conv2d(input=h_trans,
					 filter=self.W_conv1,
					 strides=[1, 1, 1, 1],
					 padding='SAME') +
			    self.b_conv1)
			
			x_vector_2 = tf.reshape(h_conv1, [-1, 7*7*512])
			with tf.variable_scope("LSTM2"):
				[h_fc_loc3, self.state2] = self.lstm2( x_vector_2, self.state2 )
			h_fc_loc3_drop = tf.nn.dropout(h_fc_loc3, self.keep_prob)
			h_fc_loc4 = tf.nn.tanh(tf.matmul(h_fc_loc3_drop, self.W_fc_loc4) + self.b_fc_loc4)
			out_size = (7, 7)
			h_trans_2 = transformer(h_conv1, h_fc_loc4, out_size)

			h_conv2 = tf.nn.relu(
			    tf.nn.conv2d(input=h_trans_2,
					 filter=self.W_conv2,
					 strides=[1, 1, 1, 1],
					 padding='SAME') +
			    self.b_conv2)

			# %% We'll now reshape so we can connect to a fully-connected layer:
			h_conv2_flat = tf.reshape(h_conv2, [-1, 7*7*512])

			h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
			h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

			with tf.variable_scope("LSTM3"):
				[output, self.state3] = self.lstm3( h_fc1_drop, self.state3 )
			output_drop = tf.nn.dropout(output, self.keep_prob)

			y_logits = tf.matmul(output_drop, self.W_fc2) + self.b_fc2

			indices = tf.cast(tf.expand_dims(tf.range(0,self.batch_size, 1), 1), tf.int32)
			concated = tf.concat(1, [indices, tf.expand_dims(y[:,time],1)])
			y_target = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.dataset.n_classes]), 1.0, 0.0)
			cross_entropy.append(tf.reduce_mean(
		    		tf.nn.softmax_cross_entropy_with_logits(y_logits, y_target)))

		cross_entropy = tf.reduce_mean(tf.pack(cross_entropy))
		# %% Monitor accuracy
		correct_prediction = tf.equal(tf.cast(tf.argmax(y_logits, 1),tf.int32), y[:,self.n_times-self.start_time-1])
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		return x, y, cross_entropy, accuracy, pool_5, boxes, scores
