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

	def train(self):
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.dataset.get_batch()
		iter_per_epoch = len(self.dataset.train_batch)
		test_iter_per_epoch = len(self.dataset.test_batch)
		[x, y, cross_entropy, accuracy, batch_pool_5_1, boxes, scores] = self.model.model(sess)

		saver = tf.train.Saver(max_to_keep=100)
		if not os.path.isdir(self.ck_dir):
			call('mkdir '+self.ck_dir,shell=True)

		# %% Define loss/eval/training functions
		opt = tf.train.AdamOptimizer(learning_rate=0.0001)
		optimizer = opt.minimize(cross_entropy)
		grads = opt.compute_gradients(cross_entropy, model.variables)

		# %% We now create a new session to actually perform the initialization the
		# variables:
		sess.run(tf.initialize_all_variables())
		#self.model.faster_rcnn.load(self.detector_dir,sess)

		for epoch_i in range(self.n_epochs):
		    np.random.shuffle(self.dataset.train_batch)
		    for iter_i in range(iter_per_epoch):
			[batch_xs, batch_ys, batch_anno] = self.dataset.get_data('train', iter_i)
			#batch_ys = dense_to_one_hot(batch_ys, n_classes=self.dataset.n_classes)

			[_, loss, tmp_batch_pool_5_1, tmp_boxes, tmp_scores] = sess.run([optimizer, cross_entropy, batch_pool_5_1, boxes, scores], feed_dict={
			    x: batch_xs[:,self.model.start_time:,:,:,:], y: batch_ys[:,self.model.start_time:]})
			print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
			pdb.set_trace()
		
		    acc = []
		    for iter_i in range(test_iter_per_epoch):
			[batch_xs, batch_ys, batch_anno] = self.dataset.get_data('test', iter_i)
			acc.append(sess.run(accuracy,feed_dict={x: batch_xs[:,self.model.start_time:,:,:,:],y: batch_ys[:,self.model.start_time:]}))
			print iter_i, "Acc:", acc[-1]
		    print('Accuracy (%d): ' % epoch_i + str(np.mean(acc)))
		    saver.save(sess, os.path.join(self.ck_dir, 'model'), global_step=epoch_i)

if __name__ == "__main__":
	sys.path.insert(0, 'data/dashcam/')
	from dashcam import dashcam
	ds = dashcam("label", "Videos", 10, main_dir='data/dashcam/')
	model = ant(ds,10,100,start_time=95, device = [0,2])
	train = trainer(model, ds, 1, 'VGGnet_fast_rcnn_iter_70000.ckpt.npy', device = [0,2])
	with tf.device('/gpu:'+str(2)):
		train.train()
