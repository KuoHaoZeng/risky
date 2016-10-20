import _init_paths
import fast_rcnn.test
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import tensorflow as tf
from spatial_transformer import transformer
from threading import Thread
import numpy as np
import multiprocessing as mp
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from ant import ant
from ant_1dim import ant_1dim
from subprocess import call
import sys, os, pdb, time

blob_shape = [10, 100, 562, 1000, 3]

def get_blobs(imgs):
	datas = np.zeros((blob_shape[1], blob_shape[2], blob_shape[3], blob_shape[4]))
	im_scales = np.zeros((blob_shape[1]))
	for time_idx in xrange(imgs.shape[0]):
		blob, im_scale = fast_rcnn.test._get_blobs(imgs[time_idx,:,:,:], None)
		datas[time_idx,:,:,:] = blob['data'][0,:,:,:]
		im_scales[time_idx] = im_scale[0]
	return datas, im_scales

def _get_blobs_par(imgs):
	blob, im_scale = fast_rcnn.test._get_blobs(imgs, None)
	return blob['data'][0,:,:,:], im_scale[0]

class trainer(object):
	def __init__(self, model, dataset, n_epochs, detector_dir, device = [0,1,2], ck_dir = 'check_points'):
		self.model = model
		self.dataset = dataset
		self.n_epochs = n_epochs
		self.detector_dir = detector_dir
		self.ck_dir = ck_dir
		self.device = device

	def get_rpn_input(self, imgs, anno=None, vid=None):
		time_a = time.time()
		blob, im_scale = fast_rcnn.test._get_blobs(imgs[0,0,:,:,:], None)
		datas = np.zeros((imgs.shape[0],imgs.shape[1],blob['data'].shape[1],blob['data'].shape[2],blob['data'].shape[3]))
		im_scales = np.zeros(imgs.shape[0:2])
		gt_boxes = []
		print('		Time for allocating memory:' + str(time.time() - time_a))
		time_a = time.time()
		for batch_idx in xrange(imgs.shape[0]):
			"""
			time_b = time.time()
			p = mp.Pool(10)
			data = p.map(_get_blobs_par, imgs[batch_idx,:,:,:,:])
			p.close()
			print('         	Time for using faster rcnn package:' + str(time.time() - time_b))
			time_b = time.time()
			datas[batch_idx,:,:,:,:] = np.array([ele[0] for ele in data])
			im_scales[batch_idx,:] = np.array([ele[1] for ele in data])
			print('         	Time for using numpy:' + str(time.time() - time_b))
			"""
			gt_boxes_batch = []
			for time_idx in xrange(imgs.shape[1]):
				#time_b = time.time()
				blob, im_scale = fast_rcnn.test._get_blobs(imgs[batch_idx,time_idx,:,:,:], None)
				#print('         	Time for using faster rcnn package:' + str(time.time() - time_b))
				#time_b = time.time()
				datas[batch_idx, time_idx, :, :, :] = blob['data'][0,:,:,:]
				#print('         	Time for using getting images:' + str(time.time() - time_b))
				#time_b = time.time()
				im_scales[batch_idx, time_idx] = im_scale[0]
				#print('         	Time for using getting scale information:' + str(time.time() - time_b))
				if anno != None:
					gt_boxes_time = []
					label_idx = 0
					for k, v in anno[batch_idx].items():
						ratio_h = ds.label[vid[batch_idx]]['size'][0] / float(blob_shape[2])
						ratio_w = ds.label[vid[batch_idx]]['size'][1] / float(blob_shape[3])
						boxes = []
						for iii in xrange(4):
							if iii % 2 == 0:
								boxes.append(v['bbox'][time_idx][iii] / ratio_w)
							else:
								boxes.append(v['bbox'][time_idx][iii] / ratio_h)
						gt_boxes_time.append(boxes+[label_idx])
						label_idx += 1
					if len(gt_boxes_time) == 0:
						gt_boxes_time.append([0,0,0,0,0])
						gt_boxes_time.append([0,0,0,0,0])
					gt_boxes_batch.append(gt_boxes_time)
			gt_boxes.append(gt_boxes_batch)
		print('		Time for getting blob:' + str(time.time() - time_a))
		
		"""
		time_a = time.time()
		p = mp.Pool(10)
		data = p.map(get_blobs, imgs)
		p.close()
		datas = np.array([ele[0] for ele in data])
		im_scales = np.array([ele[1] for ele in data])
		print('		Time for getting blob:' + str(time.time() - time_a))
		"""
		im_info = np.array([[datas.shape[2], datas.shape[3], im_scales[0,0]]]*imgs.shape[1],dtype=np.float32)
		if anno == None:
			return datas, im_scales, im_info
		else:
			return datas, im_scales, im_info, gt_boxes

	def extract_feature(self, is_gt = False):
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.dataset.get_batch()
		iter_per_epoch = len(self.dataset.train_batch)
		test_iter_per_epoch = len(self.dataset.test_batch)
		if is_gt:
			[data, im_info, gt_boxes, pool_5_tf, fc7_tf, rois_tf, bbox_pred_tf, cls_prob_tf, cls_score_tf] = self.model.faster_rcnn.extract_gt()
		else:
			[data, im_info, pool_5_tf, fc7_tf, rois_tf, bbox_pred_tf, cls_prob_tf, cls_score_tf] = self.model.faster_rcnn.extract()
		sess.run(tf.initialize_all_variables())
		self.dataset.split = 'test'
		for iter_i in range(int(test_iter_per_epoch*0.5),int(test_iter_per_epoch)):
			time_a = time.time()
			[batch_xs, batch_ys, batch_anno, vid] = self.dataset.get_data(iter_i)
			#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
			print('Time for loading data:' + str(time.time() - time_a))
			time_a = time.time()
			if is_gt:
				[batch_rpn, batch_rpn_scales, batch_rpn_info, batch_gt_boxes] = self.get_rpn_input(batch_xs, anno=batch_anno, vid=vid)
			else:
				[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)
			print('Time for preprocessing data:' + str(time.time() - time_a))

			time_a = time.time()
			boxes_batch = []
			scores_batch = []
			pred_boxes_batch = []
			fc7_batch = []
			agent_label_batch = []
			#pool_5_batch = []
			for batch_idx in xrange(batch_rpn.shape[0]):
				boxes_time = []
				scores_time = []
				pred_boxes_time = []
				fc7_time = []
				agent_label_time = []
				#pool_5_time = []
				for time_idx in xrange(batch_rpn.shape[1]):
					if is_gt:
						[pool_5, fc7, rois, bbox_pred, cls_prob, cls_score] = sess.run([pool_5_tf, fc7_tf, rois_tf, bbox_pred_tf, cls_prob_tf, cls_score_tf], feed_dict={data: np.expand_dims(batch_rpn[batch_idx,time_idx,:,:,:],0), im_info:batch_rpn_info, gt_boxes:batch_gt_boxes[batch_idx][time_idx]})
						agent_label_time.append(rois[1])
						rois = rois[0]
					else:
						[pool_5, fc7, rois, bbox_pred, cls_prob, cls_score] = sess.run([pool_5_tf, fc7_tf, rois_tf, bbox_pred_tf, cls_prob_tf, cls_score_tf], feed_dict={data: np.expand_dims(batch_rpn[batch_idx,time_idx,:,:,:],0), im_info:batch_rpn_info})
					boxes = rois[:, 1:5] / batch_rpn_scales[batch_idx,time_idx]
					scores = cls_prob
					box_deltas = bbox_pred
					pred_boxes = bbox_transform_inv(boxes, box_deltas)
					pred_boxes = fast_rcnn.test._clip_boxes(pred_boxes, batch_rpn.shape[2:])
					boxes_time.append(boxes)
					scores_time.append(scores)
					pred_boxes_time.append(pred_boxes)
					fc7_time.append(fc7)
					#pool_5_time.append(pool_5)
				boxes_batch.append(boxes_time)
				scores_batch.append(scores_time)
				pred_boxes_batch.append(pred_boxes_time)
				fc7_batch.append(fc7_time)
				agent_label_batch.append(agent_label_time)
				#pool_5_batch.append(pool_5_time)
			if is_gt:
				np.savez('/home/Hao/tik/risky/ant/data/dashcam/feature/test_gt/'+self.dataset.test_batch[iter_i].split('/')[-1],fc7=fc7_batch,boxes=boxes_batch,scores=scores_batch,pred_boxes=pred_boxes_batch, agent = agent_label_batch)
			else:
				np.savez('/home/Hao/tik/risky/ant/data/dashcam/feature/test/'+self.dataset.test_batch[iter_i].split('/')[-1],fc7=fc7_batch,boxes=boxes_batch,scores=scores_batch,pred_boxes=pred_boxes_batch)
			print('Time for feedforwarding:' + str(time.time() - time_a))

	def train(self):
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		self.dataset.get_batch()
		iter_per_epoch = len(self.dataset.train_batch)
		#iter_per_epoch = 0
		test_iter_per_epoch = len(self.dataset.test_batch)
		[x, y, rpn_info, cross_entropy, accuracy, batch_pool_5_1] = self.model.model(sess)

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
		    self.dataset.split = 'train'
		    # Use queue to load batch
		    #self.dataset.data_thread = Thread(target=self.dataset.load_data_into_queue)
		    #self.dataset.data_thread.start()
		    for iter_i in range(iter_per_epoch):
			time_a = time.time()
			[batch_xs, batch_ys, batch_anno, vid] = self.dataset.get_data(iter_i)

			print('Time for loading data:' + str(time.time() - time_a))
			#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
			#time_a = time.time()
			#[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)
			#print('Time for preprocessing data:' + str(time.time() - time_a))

			time_a = time.time()
			[_, loss, tmp_batch_pool_5_1] = sess.run([optimizer, cross_entropy, batch_pool_5_1], feed_dict={
			    x: batch_xs[:,self.model.start_time:,:,:,:], y: batch_ys[:,self.model.start_time:], rpn_info:np.array([[360,640,1]])})
			print('Time for training:' + str(time.time() - time_a))
			print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
		    #self.dataset.data_thread.join()

		    self.dataset.split = 'test'
		    # Use queue to load batch
		    #self.dataset.data_thread = Thread(target=self.dataset.load_data_into_queue)
		    #self.dataset.data_thread.start()
		    acc = []
		    for iter_i in range(test_iter_per_epoch):
			[batch_xs, batch_ys, batch_anno] = self.dataset.get_data(iter_i)
			#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
			#[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)

			acc.append(sess.run(accuracy,feed_dict={x: batch_xs[:,self.model.start_time:,:,:,:],y: batch_ys[:,self.model.start_time:], rpn_info:np.array([[360,640,1]])}))
			print iter_i, "Acc:", np.mean(acc[-1])
		    print('Accuracy (%d): ' % epoch_i + str(np.mean(np.array(acc).flatten())))
		    saver.save(sess, os.path.join(self.ck_dir, 'model'), global_step=epoch_i)
		    #self.dataset.data_thread.join()

if __name__ == "__main__":
	sys.path.insert(0, 'data/dashcam/')
	from dashcam import dashcam
	ds = dashcam("label", "Videos", 10, main_dir='data/dashcam/', workers = 1)
	model = ant_1dim(ds, 10, 100,'dashcam_model.npy',start_time=50, device = [0,1,2,3])
	train = trainer(model, ds, 1, 'dashcam_model.npy', device = [0,1,2,3])
	with tf.device('/gpu:'+str(0)):
		#train.train()
		#train.extract_feature(True)
		train.extract_feature()
