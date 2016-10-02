import tensorflow as tf
import numpy as np
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
import pdb

DEFAULT_PADDING = 'SAME'

class faster_rcnn_tf(object):
	def __init__(self, model_dir, dropout_rate):
		self.model_dir = model_dir
		self.anchor_scales = [8, 16, 32]
		self._feat_stride = [16,]
		self.dropout_rate = dropout_rate
		self.load_model()

	def load_model(self):
		model = np.load(self.model_dir).item()
		self.W_conv1_1 = tf.Variable(model['conv1_1']['weights'])
		self.b_conv1_1 = tf.Variable(model['conv1_1']['biases'])
		self.W_conv1_2 = tf.Variable(model['conv1_2']['weights'])
		self.b_conv1_2 = tf.Variable(model['conv1_2']['biases'])
		self.W_conv2_1 = tf.Variable(model['conv2_1']['weights'])
		self.b_conv2_1 = tf.Variable(model['conv2_1']['biases'])
		self.W_conv2_2 = tf.Variable(model['conv2_2']['weights'])
		self.b_conv2_2 = tf.Variable(model['conv2_2']['biases'])
		self.W_conv3_1 = tf.Variable(model['conv3_1']['weights'])
		self.b_conv3_1 = tf.Variable(model['conv3_1']['biases'])
		self.W_conv3_2 = tf.Variable(model['conv3_2']['weights'])
		self.b_conv3_2 = tf.Variable(model['conv3_2']['biases'])
		self.W_conv3_3 = tf.Variable(model['conv3_3']['weights'])
		self.b_conv3_3 = tf.Variable(model['conv3_3']['biases'])
		self.W_conv4_1 = tf.Variable(model['conv4_1']['weights'])
		self.b_conv4_1 = tf.Variable(model['conv4_1']['biases'])
		self.W_conv4_2 = tf.Variable(model['conv4_2']['weights'])
		self.b_conv4_2 = tf.Variable(model['conv4_2']['biases'])
		self.W_conv4_3 = tf.Variable(model['conv4_3']['weights'])
		self.b_conv4_3 = tf.Variable(model['conv4_3']['biases'])
		self.W_conv5_1 = tf.Variable(model['conv5_1']['weights'])
		self.b_conv5_1 = tf.Variable(model['conv5_1']['biases'])
		self.W_conv5_2 = tf.Variable(model['conv5_2']['weights'])
		self.b_conv5_2 = tf.Variable(model['conv5_2']['biases'])
		self.W_conv5_3 = tf.Variable(model['conv5_3']['weights'])
		self.b_conv5_3 = tf.Variable(model['conv5_3']['biases'])
		self.W_rpn_conv = tf.Variable(model['rpn_conv']['3x3/weights'])
		self.b_rpn_conv = tf.Variable(model['rpn_conv']['3x3/biases'])
		self.W_rpn_cls_score = tf.Variable(model['rpn_cls_score']['weights'])
		self.b_rpn_cls_score = tf.Variable(model['rpn_cls_score']['biases'])
		self.W_rpn_bbox_pred = tf.Variable(model['rpn_bbox_pred']['weights'])
		self.b_rpn_bbox_pred = tf.Variable(model['rpn_bbox_pred']['biases'])

		self.W_bbox_pred = tf.Variable(model['bbox_pred']['weights'])
		self.b_bbox_pred = tf.Variable(model['bbox_pred']['biases'])
		self.W_fc6 = tf.Variable(model['fc6']['weights'])
		self.b_fc6 = tf.Variable(model['fc6']['biases'])
		self.W_fc7 = tf.Variable(model['fc7']['weights'])
		self.b_fc7 = tf.Variable(model['fc7']['biases'])
		self.W_cls_score = tf.Variable(model['cls_score']['weights'])
		self.b_cls_score = tf.Variable(model['cls_score']['biases'])

	def run(self, data, im_info):
		h1_1 = tf.nn.relu(tf.nn.conv2d(input=data,filter=self.W_conv1_1,strides=[1, 1, 1, 1],padding='SAME', name='conv1_1') + self.b_conv1_1)
		h1_2 = tf.nn.relu(tf.nn.conv2d(input=h1_1,filter=self.W_conv1_2,strides=[1, 1, 1, 1],padding='SAME', name='conv1_2') + self.b_conv1_2)
		h1_max = self.max_pool(h1_2, 2, 2, 2, 2, padding='VALID', name='pool1')
		h2_1 = tf.nn.relu(tf.nn.conv2d(input=h1_max,filter=self.W_conv2_1,strides=[1, 1, 1, 1],padding='SAME', name='conv2_1') + self.b_conv2_1)
		h2_2 = tf.nn.relu(tf.nn.conv2d(input=h2_1,filter=self.W_conv2_2,strides=[1, 1, 1, 1],padding='SAME', name='conv2_2') + self.b_conv2_2)
		h2_max = self.max_pool(h2_2, 2, 2, 2, 2, padding='VALID', name='pool2')
		h3_1 = tf.nn.relu(tf.nn.conv2d(input=h2_max,filter=self.W_conv3_1,strides=[1, 1, 1, 1],padding='SAME', name='conv3_1') + self.b_conv3_1)
		h3_2 = tf.nn.relu(tf.nn.conv2d(input=h3_1,filter=self.W_conv3_2,strides=[1, 1, 1, 1],padding='SAME', name='conv3_2') + self.b_conv3_2)
		h3_3 = tf.nn.relu(tf.nn.conv2d(input=h3_2,filter=self.W_conv3_3,strides=[1, 1, 1, 1],padding='SAME', name='conv3_3') + self.b_conv3_3)
		h3_max = self.max_pool(h3_3, 2, 2, 2, 2, padding='VALID', name='pool3')
		h4_1 = tf.nn.relu(tf.nn.conv2d(input=h3_max,filter=self.W_conv4_1,strides=[1, 1, 1, 1],padding='SAME', name='conv4_1') + self.b_conv4_1)
		h4_2 = tf.nn.relu(tf.nn.conv2d(input=h4_1,filter=self.W_conv4_2,strides=[1, 1, 1, 1],padding='SAME', name='conv4_2') + self.b_conv4_2)
		h4_3 = tf.nn.relu(tf.nn.conv2d(input=h4_2,filter=self.W_conv4_3,strides=[1, 1, 1, 1],padding='SAME', name='conv4_3') + self.b_conv4_3)
		h4_max = self.max_pool(h4_3, 2, 2, 2, 2, padding='VALID', name='pool4')
		h5_1 = tf.nn.relu(tf.nn.conv2d(input=h4_max,filter=self.W_conv5_1,strides=[1, 1, 1, 1],padding='SAME', name='conv5_1') + self.b_conv5_1)
		h5_2 = tf.nn.relu(tf.nn.conv2d(input=h5_1,filter=self.W_conv5_2,strides=[1, 1, 1, 1],padding='SAME', name='conv5_2') + self.b_conv5_2)
		h5_3 = tf.nn.relu(tf.nn.conv2d(input=h5_2,filter=self.W_conv5_3,strides=[1, 1, 1, 1],padding='SAME', name='conv5_3') + self.b_conv5_3)

		h_rpn = tf.nn.relu(tf.nn.conv2d(input=h5_3,filter=self.W_rpn_conv,strides=[1, 1, 1, 1],padding='SAME', name='rpn_conv/3x3') + self.b_rpn_conv)
		h_rpn_cls_score = tf.nn.conv2d(input=h_rpn,filter=self.W_rpn_cls_score,strides=[1, 1, 1, 1],padding='VALID', name='rpn_cls_score') + self.b_rpn_cls_score
		h_rpn_bbox_pred = tf.nn.conv2d(input=h_rpn,filter=self.W_rpn_bbox_pred,strides=[1, 1, 1, 1],padding='VALID', name='rpn_bbox_pred') + self.b_rpn_bbox_pred
		h_rpn_cls_score_reshape = self.reshape_layer(h_rpn_cls_score, 2, name='rpn_cls_score_reshape')
		pdb.set_trace()
		h_rpn_cls_prob = self.softmax(h_rpn_cls_score_reshape, name='rpn_cls_prob')
		h_rpn_cls_prob_reshape = self.reshape_layer(h_rpn_cls_prob, len(self.anchor_scales)*3*2, name='rpn_cls_prob_reshape')

		rois = self.proposal_layer([h_rpn_cls_prob_reshape,h_rpn_bbox_pred,im_info],self._feat_stride, self.anchor_scales, name = 'rois')
		pool_5 = self.roi_pool([h5_3, rois], 7, 7, 1.0/16, name='pool_5')

		fc6 = tf.nn.relu_layer(tf.reshape(tf.transpose(pool_5,[0,3,1,2]), [-1, 7*7*512]), self.W_fc6, self.b_fc6, name='fc6')
		fc6_drop = self.dropout(fc6, self.dropout_rate, name='drop6')
		fc7 = tf.nn.relu_layer(fc6_drop, self.W_fc7, self.b_fc7, name='fc7')
		fc7_drop = self.dropout(fc7, self.dropout_rate, name='drop7')
		cls_score = tf.nn.relu_layer(fc7_drop, self.W_cls_score, self.b_cls_score, name='cls_score')
		cls_prob = self.softmax(cls_score, name='cls_prob')
		bbox_pred = tf.nn.relu_layer(fc7_drop, self.W_bbox_pred, self.b_bbox_pred, name='bbox_pred')
		
		return pool_5
	
	def validate_padding(self, padding):
        	assert padding in ('SAME', 'VALID')

	def relu(self, input, name):
		return tf.nn.relu(input, name=name)

	def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.max_pool(input,
				      ksize=[1, k_h, k_w, 1],
				      strides=[1, s_h, s_w, 1],
				      padding=padding,
				      name=name)

	def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.avg_pool(input,
				      ksize=[1, k_h, k_w, 1],
				      strides=[1, s_h, s_w, 1],
				      padding=padding,
				      name=name)
	
	def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
		# only use the first input
		if isinstance(input[0], tuple):
		    input[0] = input[0][0]

		if isinstance(input[1], tuple):
		    input[1] = input[1][0]

		print input
		return roi_pool_op.roi_pool(input[0], input[1],
					    pooled_height,
					    pooled_width,
					    spatial_scale,
					    name=name)[0]

	def proposal_layer(self, input, _feat_stride, anchor_scales, name):
		if isinstance(input[0], tuple):
		    input[0] = input[0][0]
		return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)
		
	def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
		if isinstance(input[0], tuple):
		    input[0] = input[0][0]

		with tf.variable_scope(name) as scope:

		    rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

		    rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
		    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
		    rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
		    rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


		return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
	
	def proposal_target_layer(self, input, classes, name):
		if isinstance(input[0], tuple):
		    input[0] = input[0][0]
		with tf.variable_scope(name) as scope:

		    rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

		    rois = tf.reshape(rois,[-1,5] , name = 'rois') 
		    labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
		    bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
		    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
		    bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

		   
		    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

	def reshape_layer(self, input, d,name):
		#input_shape = input.get_shape()
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob_reshape':
		     #return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[-1,d,int(input_shape[1].value/float(d)*input_shape[3].value),input_shape[2].value]),[0,2,3,1],name=name)
		     return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
			    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
		else:
		     #return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[-1,d,input_shape[1].value*(input_shape[3].value/d),input_shape[2].value]),[0,2,3,1],name=name)
		     return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
			    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

	def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
		return feature_extrapolating_op.feature_extrapolating(input,
				      scales_base,
				      num_scale_base,
				      num_per_octave,
				      name=name)

	def lrn(self, input, radius, alpha, beta, name, bias=1.0):
		return tf.nn.local_response_normalization(input,
							  depth_radius=radius,
							  alpha=alpha,
							  beta=beta,
							  bias=bias,
							  name=name)

	def concat(self, inputs, axis, name):
		return tf.concat(concat_dim=axis, values=inputs, name=name)

	def softmax(self, input, name):
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob':
		    return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
		else:
		    return tf.nn.softmax(input,name=name)

	def dropout(self, input, keep_prob, name):
		return tf.nn.dropout(input, keep_prob, name=name)
