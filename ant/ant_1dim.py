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

Dims = 256

class ant_1dim(object):
	def __init__(self, dataset, batch_size, n_times, faster_rcnn_model, memory_slots_num = 4, keep_prob = 0.9, start_time = 0, device = [0,1,2]):
		self.variables = []
		self.batch_size = batch_size
		self.start_time = start_time
		self.n_times = n_times
		self.keep_prob = keep_prob
		#self.faster_rcnn = get_network('VGGnet_test')
		self.faster_rcnn = faster_rcnn_tf(faster_rcnn_model, 1)
		self.model_variable()
		self.dataset = dataset
		self.device = device
		self.memory_slots_num = memory_slots_num

	def model_variable(self):

		# environment input soft-attention for environment stream
		self.W_conv_chr_mn = weight_variable([Dims, Dims])
		self.b_conv_chr_mn = bias_variable([Dims])
		self.variables.append(self.W_conv_chr_mn)
		self.variables.append(self.b_conv_chr_mn)

		self.W_conv_cha_mn = weight_variable([Dims, Dims])
		self.variables.append(self.W_conv_cha_mn)

		self.W_conv_cxr_mn = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_cxr_mn)

		self.W_conv_cxa_mn = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_cxa_mn)

		self.W_conv_c_mn = weight_variable([Dims, 1])
		self.variables.append(self.W_conv_c_mn)

		# input gate
		# environment part
		self.W_conv_icha = weight_variable([Dims, Dims])
		self.b_conv_icha = bias_variable([Dims])
		self.variables.append(self.W_conv_icha)
		self.variables.append(self.b_conv_icha)

		self.W_conv_ichr = weight_variable([Dims, Dims])
		self.variables.append(self.W_conv_ichr)

		self.W_conv_icxa = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_icxa)

		self.W_conv_icxr = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_icxr)

		# agent part
		self.W_fc_ifha = weight_variable([Dims, Dims])
		self.b_fc_ifha = bias_variable([Dims])
		self.variables.append(self.W_fc_ifha)
		self.variables.append(self.b_fc_ifha)

		self.W_fc_ifhr = weight_variable([Dims, Dims])
		self.variables.append(self.W_fc_ifhr)

		self.W_fc_ifxa = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ifxa)

		self.W_fc_ifxr = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ifxr)

		# forget gate
		# environment part
		self.W_conv_fcha = weight_variable([Dims, Dims])
		self.b_conv_fcha = bias_variable([Dims])
		self.variables.append(self.W_conv_fcha)
		self.variables.append(self.b_conv_fcha)

		self.W_conv_fchr = weight_variable([Dims, Dims])
		self.variables.append(self.W_conv_fchr)

		self.W_conv_fcxa = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_fcxa)

		self.W_conv_fcxr = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_fcxr)

		# agent part
		self.W_fc_ffha = weight_variable([Dims, Dims])
		self.b_fc_ffha = bias_variable([Dims])
		self.variables.append(self.W_fc_ffha)
		self.variables.append(self.b_fc_ffha)

		self.W_fc_ffhr = weight_variable([Dims, Dims])
		self.variables.append(self.W_fc_ffhr)

		self.W_fc_ffxa = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ffxa)

		self.W_fc_ffxr = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ffxr)

		# output gate
		# environment part
		self.W_conv_ocha = weight_variable([Dims, Dims])
		self.b_conv_ocha = bias_variable([Dims])
		self.variables.append(self.W_conv_ocha)
		self.variables.append(self.b_conv_ocha)

		self.W_conv_ochr = weight_variable([Dims, Dims])
		self.variables.append(self.W_conv_ochr)

		self.W_conv_ocxa = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_ocxa)

		self.W_conv_ocxr = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_ocxr)

		# agent part
		self.W_fc_ofha = weight_variable([Dims, Dims])
		self.b_fc_ofha = bias_variable([Dims])
		self.variables.append(self.W_fc_ofha)
		self.variables.append(self.b_fc_ofha)

		self.W_fc_ofhr = weight_variable([Dims, Dims])
		self.variables.append(self.W_fc_ofhr)

		self.W_fc_ofxa = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ofxa)

		self.W_fc_ofxr = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_ofxr)

		# momery
		# environment part
		self.W_conv_mcha = weight_variable([Dims, Dims])
		self.b_conv_mcha = bias_variable([Dims])
		self.variables.append(self.W_conv_mcha)
		self.variables.append(self.b_conv_mcha)

		self.W_conv_mchr = weight_variable([Dims, Dims])
		self.variables.append(self.W_conv_mchr)

		self.W_conv_mcxa = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_mcxa)

		self.W_conv_mcxr = weight_variable([4096, Dims])
		self.variables.append(self.W_conv_mcxr)

		# agent part
		self.W_fc_mfha = weight_variable([Dims, Dims])
		self.b_fc_mfha = bias_variable([Dims])
		self.variables.append(self.W_fc_mfha)
		self.variables.append(self.b_fc_mfha)

		self.W_fc_mfhr = weight_variable([Dims, Dims])
		self.variables.append(self.W_fc_mfhr)

		self.W_fc_mfxa = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_mfxa)

		self.W_fc_mfxr = weight_variable([4096, Dims])
		self.variables.append(self.W_fc_mfxr)

		# %% Create a fully-connected layer:
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

	def model(self, sess):
		# %% Graph representation of our network

		x = tf.placeholder(tf.uint8, [None, self.n_times-self.start_time, 360, 640, 3])
		x = tf.cast(x,tf.float32)
		y = tf.placeholder(tf.int32, [None, self.n_times-self.start_time])
		#rpn = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 562, 1000, 3])
		rpn_info = tf.placeholder(tf.float32, [1,3])

		hr = tf.zeros([self.batch_size, self.memory_slots_num, Dims])
		cr = tf.zeros([self.batch_size, self.memory_slots_num, Dims])
		ha = tf.zeros([self.batch_size, Dims])
		ca = tf.zeros([self.batch_size, Dims])

		cross_entropy = []
		pred = []
		for time in xrange(0, self.n_times-self.start_time):
			if time > 0:
                		tf.get_variable_scope().reuse_variables()
		
			envir_x = []
			agent_x = []
			envir_fc7 = []
			for batch in xrange(self.batch_size):
				#self.faster_rcnn.layers['data'] = tf.expand_dims(rpn[batch, time, :, :, :],0)
				#self.faster_rcnn.layers['im_info'] = rpn_info

				#cls_score = self.faster_rcnn.get_output('cls_score')
				#cls_prob = self.faster_rcnn.get_output('cls_prob')
				#bbox_pred = self.faster_rcnn.get_output('bbox_pred')
				#rois = self.faster_rcnn.get_output('rois')
				#pool_5 = self.faster_rcnn.get_output('pool_5')
				[_, fc7, _, _, _, _] = self.faster_rcnn.run(tf.expand_dims(x[batch, time, :, :, :],0), rpn_info)
				num_boxes = tf.shape(fc7)[0]
				fc7_agent = tf.gather(fc7, num_boxes-1)
				envir_fc7.append(fc7)
				agent_x.append(fc7_agent)
				"""
				boxes = rois[:, 1:5] / im_scales[0]

				scores = cls_prob

				if cfg.TEST.BBOX_REG:
					# Apply bounding-box regression deltas
					box_deltas = bbox_pred
					pred_boxes = bbox_transform_inv(boxes, box_deltas)
					pred_boxes = _clip_boxes(pred_boxes, im.shape)
				else:
					# Simply repeat the boxes, once for each class
					pred_boxes = np.tile(boxes, (1, scores.shape[1]))
				"""
				"""
				envir_b_r_x = []
				for memory_idx in xrange(self.memory_slots_num):
					envir_conv_r = tf.tanh(
					    tf.matmul(fc7, self.W_conv_cxr_mn) + 
					    tf.matmul(tf.tile(tf.expand_dims(fc7_agent,0),[num_boxes,1]), self.W_conv_cxa_mn) + 
					    tf.matmul(tf.tile(tf.expand_dims(hr[batch,memory_idx,:],0),[num_boxes,1]), self.W_conv_chr_mn) + self.b_conv_chr_mn +
					    tf.matmul(tf.tile(tf.expand_dims(ha[batch,:],0),[num_boxes,1]), self.W_conv_cha_mn)
					)
					s_conv_r = tf.nn.softmax(tf.matmul(envir_conv_r, self.W_conv_c_mn))
					envir_b_r_x.append(tf.reduce_sum(s_conv_r * fc7, 0))
				envir_b_r_x = tf.pack(envir_b_r_x)
				envir_x.append(envir_b_r_x)
				"""

			with tf.device('/gpu:'+str(self.device[1])):
				envir_fc7 = tf.pack(envir_fc7)
				agent_x = tf.pack(agent_x)
				envir_fc7_shape = tf.shape(envir_fc7)
				envir_conv_r = tf.batch_matmul(tf.tanh(
					tf.tile(tf.expand_dims(tf.batch_matmul(envir_fc7, tf.tile(tf.expand_dims(self.W_conv_cxr_mn,0),[envir_fc7_shape[0],1,1])),1),[1,self.memory_slots_num,1,1]) +
					tf.tile(tf.expand_dims(tf.expand_dims(tf.matmul(agent_x,self.W_conv_cxr_mn),1),1),[1,self.memory_slots_num,envir_fc7_shape[1],1]) +
					tf.tile(tf.expand_dims(tf.batch_matmul(hr,tf.tile(tf.expand_dims(self.W_conv_chr_mn,0),[envir_fc7_shape[0],1,1])) + tf.tile(tf.expand_dims(tf.expand_dims(self.b_conv_chr_mn,0),0),[envir_fc7_shape[0],self.memory_slots_num,1]),2),[1,1,envir_fc7_shape[1],1]) +
					tf.tile(tf.expand_dims(tf.expand_dims(tf.matmul(ha,self.W_conv_cha_mn),1),1),[1,self.memory_slots_num,envir_fc7_shape[1],1])
				), tf.tile(tf.expand_dims(tf.expand_dims(self.W_conv_c_mn,0),0),[envir_fc7_shape[0],self.memory_slots_num,1,1]))
				s_conv_r = tf.reshape(tf.nn.softmax(tf.reshape(envir_conv_r,[envir_fc7_shape[0],self.memory_slots_num*envir_fc7_shape[1]])),[envir_fc7_shape[0],self.memory_slots_num,envir_fc7_shape[1]])
				envir_x = tf.reduce_sum(tf.tile(tf.expand_dims(s_conv_r,3),[1,1,1,4096]) * tf.tile(tf.expand_dims(envir_fc7,1),[1,self.memory_slots_num,1,1]), 2)

				#envir_x = tf.pack(envir_x)
				#agent_x = tf.pack(agent_x)

				envir_x_avg = tf.reduce_mean(envir_x,1)
				envir_h_avg = tf.reduce_mean(hr,1)

			with tf.device('/gpu:'+str(self.device[2])):
				"""
				hr_tmp = []
				cr_tmp = []
				for memory_idx in xrange(self.memory_slots_num):
					igr = tf.sigmoid(tf.matmul(hr[:,memory_idx,:], self.W_conv_ichr) + 
							tf.matmul(envir_x[:,memory_idx,:], self.W_conv_icxr) +
							tf.matmul(ha, self.W_conv_icha) + self.b_conv_icha +
							tf.matmul(agent_x, self.W_conv_icxa)
					)
					fgr = tf.sigmoid(tf.matmul(hr[:,memory_idx,:], self.W_conv_fchr) +
                                                        tf.matmul(envir_x[:,memory_idx,:], self.W_conv_fcxr) +
                                                        tf.matmul(ha, self.W_conv_fcha) + self.b_conv_fcha +
                                                        tf.matmul(agent_x, self.W_conv_fcxa)
					)
					ogr = tf.sigmoid(tf.matmul(hr[:,memory_idx,:], self.W_conv_ochr) +
                                                        tf.matmul(envir_x[:,memory_idx,:], self.W_conv_ocxr) +
                                                        tf.matmul(ha, self.W_conv_ocha) + self.b_conv_ocha +
                                                        tf.matmul(agent_x, self.W_conv_ocxa)
					)
					cr_new = tf.tanh(tf.matmul(hr[:,memory_idx,:], self.W_conv_mchr) +
                                                        tf.matmul(envir_x[:,memory_idx,:], self.W_conv_mcxr) +
                                                        tf.matmul(ha, self.W_conv_mcha) + self.b_conv_mcha +
                                                        tf.matmul(agent_x, self.W_conv_mcxa)
					)
					cr_tmp.append(fgr * cr[:,memory_idx,:] + igr * cr_new)
					hr_tmp.append(ogr * tf.tanh(cr_tmp[-1]))
				cr = tf.transpose(tf.pack(cr_tmp),[1,0,2])
				hr = tf.transpose(tf.pack(hr_tmp),[1,0,2])
				"""
				igr = tf.sigmoid(
					tf.batch_matmul(hr, tf.tile(tf.expand_dims(self.W_conv_ichr,0),[self.batch_size,1,1])) +
					tf.batch_matmul(envir_x, tf.tile(tf.expand_dims(self.W_conv_icxr,0),[self.batch_size,1,1])) +
					tf.tile(tf.expand_dims(tf.matmul(ha, self.W_conv_icha) + self.b_conv_icha,1),[1,self.memory_slots_num,1])  +
					tf.tile(tf.expand_dims(tf.matmul(agent_x, self.W_conv_icxa),1),[1,self.memory_slots_num,1])
				)
				fgr = tf.sigmoid(
					tf.batch_matmul(hr, tf.tile(tf.expand_dims(self.W_conv_fchr,0),[self.batch_size,1,1])) +
					tf.batch_matmul(envir_x, tf.tile(tf.expand_dims(self.W_conv_fcxr,0),[self.batch_size,1,1])) +
					tf.tile(tf.expand_dims(tf.matmul(ha, self.W_conv_fcha) + self.b_conv_fcha,1),[1,self.memory_slots_num,1])  +
					tf.tile(tf.expand_dims(tf.matmul(agent_x, self.W_conv_fcxa),1),[1,self.memory_slots_num,1])
				)
				ogr = tf.sigmoid(
					tf.batch_matmul(hr, tf.tile(tf.expand_dims(self.W_conv_ochr,0),[self.batch_size,1,1])) +
					tf.batch_matmul(envir_x, tf.tile(tf.expand_dims(self.W_conv_ocxr,0),[self.batch_size,1,1])) +
					tf.tile(tf.expand_dims(tf.matmul(ha, self.W_conv_ocha) + self.b_conv_ocha,1),[1,self.memory_slots_num,1])  +
					tf.tile(tf.expand_dims(tf.matmul(agent_x, self.W_conv_ocxa),1),[1,self.memory_slots_num,1])
				)
				cr_new = tf.tanh(
					tf.batch_matmul(hr, tf.tile(tf.expand_dims(self.W_conv_mchr,0),[self.batch_size,1,1])) +
					tf.batch_matmul(envir_x, tf.tile(tf.expand_dims(self.W_conv_mcxr,0),[self.batch_size,1,1])) +
					tf.tile(tf.expand_dims(tf.matmul(ha, self.W_conv_mcha) + self.b_conv_mcha,1),[1,self.memory_slots_num,1])  +
					tf.tile(tf.expand_dims(tf.matmul(agent_x, self.W_conv_mcxa),1),[1,self.memory_slots_num,1])
				)
				cr = fgr * cr + igr * cr_new
				hr = ogr * tf.tanh(cr)

			with tf.device('/gpu:'+str(self.device[2])):
				iga = tf.sigmoid(tf.matmul(envir_h_avg, self.W_fc_ifhr) + 
						tf.matmul(envir_x_avg, self.W_fc_ifxr) +
						tf.matmul(ha, self.W_fc_ifha) + self.b_fc_ifha +
						tf.matmul(agent_x, self.W_fc_ifxa)
				)
				fga = tf.sigmoid(tf.matmul(envir_h_avg, self.W_fc_ffhr) + 
						tf.matmul(envir_x_avg, self.W_fc_ffxr) +
						tf.matmul(ha, self.W_fc_ffha) + self.b_fc_ffha +
						tf.matmul(agent_x, self.W_fc_ffxa)
				)
				oga = tf.sigmoid(tf.matmul(envir_h_avg, self.W_fc_ofhr) + 
						tf.matmul(envir_x_avg, self.W_fc_ofxr) +
						tf.matmul(ha, self.W_fc_ofha) + self.b_fc_ofha +
						tf.matmul(agent_x, self.W_fc_ofxa)
				)
				ca_new = tf.tanh(tf.matmul(envir_h_avg, self.W_fc_mfhr) + 
						tf.matmul(envir_x_avg, self.W_fc_mfxr) +
						tf.matmul(ha, self.W_fc_mfha) + self.b_fc_mfha +
						tf.matmul(agent_x, self.W_fc_mfxa)
				)
				ca = fga * ca + iga * ca_new
				ha = oga * tf.tanh(ca)

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
				pred.append(y_logits[:,1])

		cross_entropy = tf.reduce_mean(tf.pack(cross_entropy))
		# %% Monitor accuracy
		pred = tf.reduce_max(tf.transpose(tf.to_int32(tf.greater_equal(tf.pack(pred),0.5))),1)
		correct_prediction = tf.equal(pred, tf.reduce_max(y,1))
		accuracy = tf.cast(correct_prediction, 'float')

		return x, y, rpn_info, cross_entropy, accuracy, fc7
