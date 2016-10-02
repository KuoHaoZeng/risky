import _init_paths
import fast_rcnn.test
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from ant import ant
from subprocess import call
import sys, os, pdb

class trainer(object):
	def __init__(self, model, dataset, n_epochs, detector_dir, device = [0,1,2], ck_dir = 'check_points'):
		self.model = model
		self.dataset = dataset
		self.n_epochs = n_epochs
		self.detector_dir = detector_dir
		self.ck_dir = ck_dir
		self.device = device

	def get_rpn_input(self, imgs):
		blob, im_scale = fast_rcnn.test._get_blobs(imgs[0,0,:,:,:], None)
		datas = np.zeros((imgs.shape[0],imgs.shape[1],blob['data'].shape[1],blob['data'].shape[2],blob['data'].shape[3]))
		im_scales = np.zeros(imgs.shape[0:2])
		for batch_idx in xrange(imgs.shape[0]):
			for time_idx in xrange(imgs.shape[1]):
				blob, im_scale = fast_rcnn.test._get_blobs(imgs[batch_idx,time_idx,:,:,:], None)
				datas[batch_idx, time_idx, :, :, :] = blob['data'][0,:,:,:]
				im_scales[batch_idx, time_idx] = im_scale[0]
		im_info = np.array([[datas.shape[2], datas.shape[3], im_scales[0,0]]],dtype=np.float32)
		return datas, im_scales, im_info

	def train(self):
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.dataset.get_batch()
		iter_per_epoch = len(self.dataset.train_batch)
		test_iter_per_epoch = len(self.dataset.test_batch)
		[x, y, rpn, rpn_info, cross_entropy, accuracy, batch_pool_5_1] = self.model.model(sess)

		saver = tf.train.Saver(max_to_keep=100)
		if not os.path.isdir(self.ck_dir):
			call('mkdir '+self.ck_dir,shell=True)

		# %% Define loss/eval/training functions
		opt = tf.train.AdamOptimizer(learning_rate=0.0001)
		#optimizer = opt.minimize(cross_entropy)
		grads = opt.compute_gradients(cross_entropy, model.variables)
		optimizer = opt.apply_gradients(grads)

		# %% We now create a new session to actually perform the initialization the
		# variables:
		sess.run(tf.initialize_all_variables())
		#self.model.faster_rcnn.load(self.detector_dir,sess)

		for epoch_i in range(self.n_epochs):
		    np.random.shuffle(self.dataset.train_batch)
		    for iter_i in range(iter_per_epoch):
			[batch_xs, batch_ys, batch_anno] = self.dataset.get_data('train', iter_i)
			[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)

			[_, loss, tmp_batch_pool_5_1] = sess.run([optimizer, cross_entropy, batch_pool_5_1], feed_dict={
			    x: batch_xs[:,self.model.start_time:,:,:,:], y: batch_ys[:,self.model.start_time:], rpn: batch_rpn[:,self.model.start_time:,:,:,:], rpn_info:batch_rpn_info})
			print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
		
		    acc = []
		    for iter_i in range(test_iter_per_epoch):
			[batch_xs, batch_ys, batch_anno] = self.dataset.get_data('test', iter_i)
			[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)

			acc.append(sess.run(accuracy,feed_dict={x: batch_xs[:,self.model.start_time:,:,:,:],y: batch_ys[:,self.model.start_time:], rpn: batch_rpn[:,self.model.start_time:,:,:,:], rpn_info:batch_rpn_info}))
			print iter_i, "Acc:", acc[-1]
		    print('Accuracy (%d): ' % epoch_i + str(np.mean(acc)))
		    saver.save(sess, os.path.join(self.ck_dir, 'model'), global_step=epoch_i)

if __name__ == "__main__":
	sys.path.insert(0, 'data/dashcam/')
	from dashcam import dashcam
	ds = dashcam("label", "Videos", 10, main_dir='data/dashcam/')
	model = ant(ds, 10, 100,'VGGnet_fast_rcnn_iter_70000.ckpt.npy',start_time=95, device = [0,1,2,3])
	train = trainer(model, ds, 1, 'VGGnet_fast_rcnn_iter_70000.ckpt.npy', device = [0,1,2,3])
	with tf.device('/gpu:'+str(0)):
		train.train()
