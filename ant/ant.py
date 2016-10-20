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

class ant(object):
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
		self.W_conv_chr_mn = weight_variable([3, 3, 512, 1])
		self.b_conv_chr_mn = bias_variable([1])
		self.variables.append(self.W_conv_chr_mn)
		self.variables.append(self.b_conv_chr_mn)

		self.W_conv_cha_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_cha_mn)

		self.W_conv_cxr_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_cxr_mn)

		self.W_conv_cxa_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_cxa_mn)

		self.W_conv_c_mn = weight_variable([3, 3, 1, 1])
		self.variables.append(self.W_conv_c_mn)
		"""
		# environment hidden soft-attention for agent stream
		self.W_conv_fhr_mn = weight_variable([2, 3, 3, 512, 1])
		self.b_conv_fhr_mn = bias_variable([1])
		self.variables.append(self.W_conv_fhr_mn)
		self.variables.append(self.b_conv_fhr_mn)

		self.W_conv_hfha_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_hfha_mn)

		self.W_conv_hfxr_mn = weight_variable([2, 3, 3, 512, 1])
		self.variables.append(self.W_conv_hfxr_mn)

		self.W_conv_hfxa_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_hfxa_mn)

		self.W_conv_hf_mn = weight_variable([3, 3, 1, 1])
		self.variables.append(self.W_conv_hf_mn)

		# environment input soft-attention for agent stream
		self.W_conv_xfhr_mn = weight_variable([3, 3, 512, 1])
		self.b_conv_xfhr_mn = bias_variable([1])
		self.variables.append(self.W_conv_xfhr_mn)
		self.variables.append(self.b_conv_xfhr_mn)

		self.W_conv_xfha_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_xfha_mn)

		self.W_conv_xfxr_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_xfxr_mn)

		self.W_conv_xfxa_mn = weight_variable([3, 3, 512, 1])
		self.variables.append(self.W_conv_xfxa_mn)

		self.W_conv_xf_mn = weight_variable([3, 3, 1, 1])
		self.variables.append(self.W_conv_xf_mn)
		"""
		# input gate
		# environment part
		self.W_conv_icha = weight_variable([3, 3, 512, 512])
		self.b_conv_icha = bias_variable([512])
		self.variables.append(self.W_conv_icha)
		self.variables.append(self.b_conv_icha)

		self.W_conv_ichr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_ichr)

		self.W_conv_icxa = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_icxa)

		self.W_conv_icxr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_icxr)

		# agent part
		self.W_fc_ifha = weight_variable([512, 512])
		self.b_fc_ifha = bias_variable([512])
		self.variables.append(self.W_fc_ifha)
		self.variables.append(self.b_fc_ifha)

		self.W_fc_ifhr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ifhr)

		self.W_fc_ifxa = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ifxa)

		self.W_fc_ifxr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ifxr)

		# forget gate
		# environment part
		self.W_conv_fcha = weight_variable([3, 3, 512, 512])
		self.b_conv_fcha = bias_variable([512])
		self.variables.append(self.W_conv_fcha)
		self.variables.append(self.b_conv_fcha)

		self.W_conv_fchr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_fchr)

		self.W_conv_fcxa = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_fcxa)

		self.W_conv_fcxr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_fcxr)

		# agent part
		self.W_fc_ffha = weight_variable([512, 512])
		self.b_fc_ffha = bias_variable([512])
		self.variables.append(self.W_fc_ffha)
		self.variables.append(self.b_fc_ffha)

		self.W_fc_ffhr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ffhr)

		self.W_fc_ffxa = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ffxa)

		self.W_fc_ffxr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ffxr)

		# output gate
		# environment part
		self.W_conv_ocha = weight_variable([3, 3, 512, 512])
		self.b_conv_ocha = bias_variable([512])
		self.variables.append(self.W_conv_ocha)
		self.variables.append(self.b_conv_ocha)

		self.W_conv_ochr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_ochr)

		self.W_conv_ocxa = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_ocxa)

		self.W_conv_ocxr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_ocxr)

		# agent part
		self.W_fc_ofha = weight_variable([512, 512])
		self.b_fc_ofha = bias_variable([512])
		self.variables.append(self.W_fc_ofha)
		self.variables.append(self.b_fc_ofha)

		self.W_fc_ofhr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ofhr)

		self.W_fc_ofxa = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ofxa)

		self.W_fc_ofxr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_ofxr)

		# momery
		# environment part
		self.W_conv_mcha = weight_variable([3, 3, 512, 512])
		self.b_conv_mcha = bias_variable([512])
		self.variables.append(self.W_conv_mcha)
		self.variables.append(self.b_conv_mcha)

		self.W_conv_mchr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_mchr)

		self.W_conv_mcxa = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_mcxa)

		self.W_conv_mcxr = weight_variable([3, 3, 512, 512])
		self.variables.append(self.W_conv_mcxr)

		# agent part
		self.W_fc_mfha = weight_variable([512, 512])
		self.b_fc_mfha = bias_variable([512])
		self.variables.append(self.W_fc_mfha)
		self.variables.append(self.b_fc_mfha)

		self.W_fc_mfhr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_mfhr)

		self.W_fc_mfxa = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_mfxa)

		self.W_fc_mfxr = weight_variable([7*7*512, 512])
		self.variables.append(self.W_fc_mfxr)

		# %% Create a fully-connected layer:
		self.W_fc1 = weight_variable([512 * 2, 4096])
		self.b_fc1 = bias_variable([4096])
		self.variables.append(self.W_fc1)
		self.variables.append(self.b_fc1)
		
		self.W_fc2 = weight_variable([4096, 4096])
		self.b_fc2 = bias_variable([4096])
		self.variables.append(self.W_fc2)
		self.variables.append(self.b_fc2)	

		self.W_fc3 = weight_variable([4096, 2])
		self.b_fc3 = bias_variable([2])
		self.variables.append(self.W_fc3)
		self.variables.append(self.b_fc3)	

	def model(self, sess):
		# %% Graph representation of our network

		x = tf.placeholder(tf.uint8, [None, self.n_times-self.start_time, 360, 640, 3])
		x = tf.cast(x,tf.float32)
		y = tf.placeholder(tf.int32, [None, self.n_times-self.start_time])
		rpn = tf.placeholder(tf.float32, [None, self.n_times-self.start_time, 562, 1000, 3])
		rpn_info = tf.placeholder(tf.float32, [1,3])

		hr = tf.zeros([self.batch_size, self.memory_slots_num, 7, 7, 512])
		cr = tf.zeros([self.batch_size, self.memory_slots_num, 7, 7, 512])
		ha = tf.zeros([self.batch_size, 512])
		ca = tf.zeros([self.batch_size, 512])

		cross_entropy = []
		for time in xrange(0, self.n_times-self.start_time):
			if time > 0:
                		tf.get_variable_scope().reuse_variables()
		
			envir_x = []
			agent_x = []
			with tf.device('/gpu:'+str(self.device[1])):
				for batch in xrange(self.batch_size):
					#self.faster_rcnn.layers['data'] = tf.expand_dims(rpn[batch, time, :, :, :],0)
					#self.faster_rcnn.layers['im_info'] = rpn_info

					#cls_score = self.faster_rcnn.get_output('cls_score')
					#cls_prob = self.faster_rcnn.get_output('cls_prob')
					#bbox_pred = self.faster_rcnn.get_output('bbox_pred')
					#rois = self.faster_rcnn.get_output('rois')
					#pool_5 = self.faster_rcnn.get_output('pool_5')
					with tf.device('/gpu:'+str(self.device[0])):
						[pool_5, _] = self.faster_rcnn.run(tf.expand_dims(rpn[batch, time, :, :, :],0), rpn_info)
						num_boxes = tf.shape(pool_5)[0]
						pool_5_agent = tf.gather(pool_5, num_boxes-1)
						agent_x.append(pool_5_agent)
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
					envir_b_r_x = []
					for memory_idx in xrange(self.memory_slots_num):
						envir_conv_r = tf.tanh(
						    tf.nn.conv2d(input=pool_5,
								 filter=self.W_conv_cxr_mn,
								 strides=[1, 1, 1, 1],
								 padding='SAME') +
						    tf.tile(tf.nn.conv2d(input=tf.expand_dims(pool_5_agent,0),
								 filter=self.W_conv_cxa_mn,
								 strides=[1, 1, 1, 1],
								 padding='SAME'),[num_boxes,1,1,1]) +
						    tf.tile(tf.nn.conv2d(input=tf.expand_dims(hr[batch,memory_idx,:,:,:],0),
								 filter=self.W_conv_chr_mn,
								 strides=[1, 1, 1, 1],
								 padding='SAME') + self.b_conv_chr_mn,[num_boxes,1,1,1]) +
						    tf.tile(tf.nn.conv2d(input=tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(ha[batch,:],0),0),[7,7,1]),0),
								 filter=self.W_conv_cha_mn,
								 strides=[1, 1, 1, 1],
								 padding='SAME'),[num_boxes,1,1,1])
						)
						s_conv_r = tf.nn.softmax(tf.nn.conv2d(input=envir_conv_r,filter=self.W_conv_c_mn,strides=[1, 1, 1, 1],padding='SAME'),0)	
						envir_b_r_x.append(tf.reduce_sum(tf.tile(s_conv_r,[1,1,1,512]) * pool_5, 0))
					envir_b_r_x = tf.pack(envir_b_r_x)
					envir_x.append(envir_b_r_x)

			envir_x = tf.pack(envir_x)
			agent_x = tf.pack(agent_x)

			envir_x_avg = tf.reduce_mean(envir_x,1)
			envir_h_avg = tf.reduce_mean(hr,1)

			with tf.device('/gpu:'+str(self.device[2])):
				hr_tmp = []
				cr_tmp = []
				for memory_idx in xrange(self.memory_slots_num):
					igr = tf.sigmoid(tf.nn.conv2d(input=hr[:,memory_idx,:,:,:],
								filter=self.W_conv_ichr,
								strides=[1, 1, 1, 1],
								padding='SAME') + 
							tf.nn.conv2d(input=envir_x[:,memory_idx,:,:,:],
								filter=self.W_conv_icxr,
								strides=[1, 1, 1, 1],
								padding='SAME') +
							tf.nn.conv2d(input=tf.tile(tf.expand_dims(tf.expand_dims(ha,1),1),[1,7,7,1]),
								filter=self.W_conv_icha,
								strides=[1, 1, 1, 1],
								padding='SAME') + self.b_conv_icha +
							tf.nn.conv2d(input=agent_x,
								filter=self.W_conv_icxa,
								strides=[1, 1, 1, 1],
								padding='SAME')
					)
					fgr = tf.sigmoid(tf.nn.conv2d(input=hr[:,memory_idx,:,:,:],
								filter=self.W_conv_fchr,
								strides=[1, 1, 1, 1],
								padding='SAME') + 
							tf.nn.conv2d(input=envir_x[:,memory_idx,:,:,:],
								filter=self.W_conv_fcxr,
								strides=[1, 1, 1, 1],
								padding='SAME') +
							tf.nn.conv2d(input=tf.tile(tf.expand_dims(tf.expand_dims(ha,1),1),[1,7,7,1]),
								filter=self.W_conv_fcha,
								strides=[1, 1, 1, 1],
								padding='SAME') + self.b_conv_fcha +
							tf.nn.conv2d(input=agent_x,
								filter=self.W_conv_fcxa,
								strides=[1, 1, 1, 1],
								padding='SAME')
					)
					ogr = tf.sigmoid(tf.nn.conv2d(input=hr[:,memory_idx,:,:,:],
								filter=self.W_conv_ochr,
								strides=[1, 1, 1, 1],
								padding='SAME') + 
							tf.nn.conv2d(input=envir_x[:,memory_idx,:,:,:],
								filter=self.W_conv_ocxr,
								strides=[1, 1, 1, 1],
								padding='SAME') +
							tf.nn.conv2d(input=tf.tile(tf.expand_dims(tf.expand_dims(ha,1),1),[1,7,7,1]),
								filter=self.W_conv_ocha,
								strides=[1, 1, 1, 1],
								padding='SAME') + self.b_conv_ocha +
							tf.nn.conv2d(input=agent_x,
								filter=self.W_conv_ocxa,
								strides=[1, 1, 1, 1],
								padding='SAME')
					)
					cr_new = tf.sigmoid(tf.nn.conv2d(input=hr[:,memory_idx,:,:,:],
								filter=self.W_conv_mchr,
								strides=[1, 1, 1, 1],
								padding='SAME') + 
							tf.nn.conv2d(input=envir_x[:,memory_idx,:,:,:],
								filter=self.W_conv_mcxr,
								strides=[1, 1, 1, 1],
								padding='SAME') +
							tf.nn.conv2d(input=tf.tile(tf.expand_dims(tf.expand_dims(ha,1),1),[1,7,7,1]),
								filter=self.W_conv_mcha,
								strides=[1, 1, 1, 1],
								padding='SAME') + self.b_conv_mcha +
							tf.nn.conv2d(input=agent_x,
								filter=self.W_conv_mcxa,
								strides=[1, 1, 1, 1],
								padding='SAME')
					)
					cr_tmp.append(fgr * cr[:,memory_idx,:,:,:] + igr * cr_new)
					hr_tmp.append(ogr * tf.tanh(cr_tmp[-1]))
				cr = tf.transpose(tf.pack(cr_tmp),[1,0,2,3,4])
				hr = tf.transpose(tf.pack(hr_tmp),[1,0,2,3,4])

			with tf.device('/gpu:'+str(self.device[3])):
				iga = tf.sigmoid(tf.matmul(tf.reshape(envir_h_avg,[self.batch_size, 7*7*512]), self.W_fc_ifhr) + 
						tf.matmul(tf.reshape(envir_x_avg,[self.batch_size, 7*7*512]), self.W_fc_ifxr) +
						tf.matmul(ha, self.W_fc_ifha) + self.b_fc_ifha +
						tf.matmul(tf.reshape(agent_x,[self.batch_size, 7*7*512]), self.W_fc_ifxa)
				)
				fga = tf.sigmoid(tf.matmul(tf.reshape(envir_h_avg,[self.batch_size, 7*7*512]), self.W_fc_ffhr) + 
						tf.matmul(tf.reshape(envir_x_avg,[self.batch_size, 7*7*512]), self.W_fc_ffxr) +
						tf.matmul(ha, self.W_fc_ffha) + self.b_fc_ffha +
						tf.matmul(tf.reshape(agent_x,[self.batch_size, 7*7*512]), self.W_fc_ffxa)
				)
				oga = tf.sigmoid(tf.matmul(tf.reshape(envir_h_avg,[self.batch_size, 7*7*512]), self.W_fc_ofhr) + 
						tf.matmul(tf.reshape(envir_x_avg,[self.batch_size, 7*7*512]), self.W_fc_ofxr) +
						tf.matmul(ha, self.W_fc_ofha) + self.b_fc_ofha +
						tf.matmul(tf.reshape(agent_x,[self.batch_size, 7*7*512]), self.W_fc_ofxa)
				)
				ca_new = tf.tanh(tf.matmul(tf.reshape(envir_h_avg,[self.batch_size, 7*7*512]), self.W_fc_mfhr) + 
						tf.matmul(tf.reshape(envir_x_avg,[self.batch_size, 7*7*512]), self.W_fc_mfxr) +
						tf.matmul(ha, self.W_fc_mfha) + self.b_fc_mfha +
						tf.matmul(tf.reshape(agent_x,[self.batch_size, 7*7*512]), self.W_fc_mfxa)
				)
				ca = fga * ca + iga * ca_new
				ha = oga * tf.tanh(ca)

			# %% We'll now reshape so we can connect to a fully-connected layer:
			h_fc1 = tf.nn.relu(tf.matmul(tf.concat(1,[ha,tf.reduce_mean(hr,[1,2,3])]), self.W_fc1) + self.b_fc1)
			h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
			h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

			y_logits = tf.matmul(h_fc2_drop, self.W_fc3) + self.b_fc3

			indices = tf.cast(tf.expand_dims(tf.range(0,self.batch_size, 1), 1), tf.int32)
			concated = tf.concat(1, [indices, tf.expand_dims(y[:,time],1)])
			y_target = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.dataset.n_classes]), 1.0, 0.0)
			cross_entropy.append(tf.reduce_mean(
		    		tf.nn.softmax_cross_entropy_with_logits(y_logits, y_target)))

		cross_entropy = tf.reduce_mean(tf.pack(cross_entropy))
		# %% Monitor accuracy
		correct_prediction = tf.equal(tf.cast(tf.argmax(y_logits, 1),tf.int32), y[:,self.n_times-self.start_time-1])
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		return x, y, rpn, rpn_info, cross_entropy, accuracy, pool_5
