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
from ant_1dim_loc import ant_1dim_loc
from subprocess import call
import sys, os, pdb, time, random
from sklearn.metrics import average_precision_score

blob_shape = [10, 100, 562, 1000, 3]

select_idx = None
batch_xs_a = None
batch_boxes_a =None
batch_boxes_ys =None
batch_boxes_a_ys = None
batch_agent = None

def select_a_time(time):
	global select_idx, batch_xs_a, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_agent
	xs_a = np.zeros((2,4096))
	boxes_a = np.zeros((2,4))
	boxes_label = np.zeros(31)
	anno_boxes_id = np.where(batch_boxes_a_ys[select_idx,time] != None)[0]
	idy = random.choice(anno_boxes_id)
	agent= batch_agent[select_idx,time][idy]
	other = np.where(np.array(batch_agent[select_idx,time])!=agent)[0]
	if len(other)!=0:
		idz = random.choice(other)
	else:
		idz = idy
	risky = batch_agent[select_idx,time][idz]
	xs_a[0,:] = batch_xs_a[select_idx,time][idy,:]
	xs_a[1,:] = batch_xs_a[select_idx,time][idz,:]
	boxes_a[0,:] = batch_boxes_a[select_idx,time][idy,:]
	boxes_a[1,:] = batch_boxes_a[select_idx,time][idz,:]
	boxes_label[np.where(batch_boxes_ys[select_idx,time,:]==risky)[0]] = 1
	boxes_label[np.where(batch_boxes_ys[select_idx,time,:]!=risky)[0]] = 0
	boxes_label[np.where(batch_boxes_ys[select_idx,time,:]==agent)[0]] = 1
	boxes_label[-1] = 1
	return xs_a, boxes_a, boxes_label

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
					for k, v in anno[batch_idx].items():
						ratio_h = ds.label[vid[batch_idx]]['size'][0] / float(blob_shape[2])
						ratio_w = ds.label[vid[batch_idx]]['size'][1] / float(blob_shape[3])
						boxes = []
						for iii in xrange(4):
							if iii % 2 == 0:
								boxes.append(v['bbox'][time_idx][iii] / ratio_w)
							else:
								boxes.append(v['bbox'][time_idx][iii] / ratio_h)
						gt_boxes_time.append(boxes+[0])
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
		self.dataset.split = 'train'
		for iter_i in range(int(iter_per_epoch/4)):
			time_a = time.time()
			[batch_xs, batch_ys, batch_anno, vid] = self.dataset.get_data(iter_i)
			#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
			print('Time for loading data:' + str(time.time() - time_a))
			time_a = time.time()
			if is_gt:
				[batch_rpn, batch_rpn_scales, batch_rpn_info, batch_gt_boxes] = self.get_rpn_input(batch_xs, anno=batch_anno, vid=vid)
			else:
				[batch_rpn, batch_rpn_scales, batch_rpn_info, batch_gt_boxes] = self.get_rpn_input(batch_xs)
			print('Time for preprocessing data:' + str(time.time() - time_a))

			time_a = time.time()
			boxes_batch = []
			scores_batch = []
			pred_boxes_batch = []
			fc7_batch = []
			#pool_5_batch = []
			for batch_idx in xrange(batch_rpn.shape[0]):
				boxes_time = []
				scores_time = []
				pred_boxes_time = []
				fc7_time = []
				#pool_5_time = []
				for time_idx in xrange(batch_rpn.shape[1]):
					if is_gt:
						[pool_5, fc7, rois, bbox_pred, cls_prob, cls_score] = sess.run([pool_5_tf, fc7_tf, rois_tf, bbox_pred_tf, cls_prob_tf, cls_score_tf], feed_dict={data: np.expand_dims(batch_rpn[batch_idx,time_idx,:,:,:],0), im_info:batch_rpn_info, gt_boxes:batch_gt_boxes[batch_idx][time_idx]})
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
				#pool_5_batch.append(pool_5_time)
			np.savez('/home/Hao/tik/risky/ant/data/dashcam/feature/train_gt/'+self.dataset.train_batch[iter_i].split('/')[-1],fc7=fc7_batch,boxes=boxes_batch,scores=scores_batch,pred_boxes=pred_boxes_batch)
			print('Time for feedforwarding:' + str(time.time() - time_a))

	def train(self):
		global batch_xs_a, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_agent
		def select_a():
			global select_idx
			xs_a = np.zeros([self.model.batch_size,self.model.n_times,2,batch_xs_a[0,0].shape[1]])
			boxes_a = np.zeros([self.model.batch_size,self.model.n_times,2,batch_boxes_a[0,0].shape[1]])
			boxes_label = np.zeros([self.model.batch_size,self.model.n_times,31])
			for select_idx in xrange(self.model.batch_size):
				if batch_ys[select_idx,0] == 1:
					"""
					# Why the parallel version isn't fater?
					p = mp.Pool(8)
                                	xxx = p.map(select_a_time, range(100))
					p.close()
					xs_a[select_idx,:,:,:] = [yyy[0] for yyy in xxx]
					boxes_a[select_idx,:,:,:] = [yyy[1] for yyy in xxx]
					boxes_label[select_idx,:,:] = [yyy[2] for yyy in xxx]
					"""
					for time in xrange(100):
						anno_boxes_id = np.where(batch_boxes_a_ys[select_idx,time] != None)[0]
						idy = random.choice(anno_boxes_id)
						agent= batch_agent[select_idx,time][idy]
						other = np.where(np.array(batch_agent[select_idx,time])!=agent)[0]
						if len(other)!=0:
							idz = random.choice(other)
						else:
							idz = idy
						risky = batch_agent[select_idx,time][idz]
						xs_a[select_idx,time,0,:] = batch_xs_a[select_idx,time][idy,:]
						xs_a[select_idx,time,1,:] = batch_xs_a[select_idx,time][idz,:]
						boxes_a[select_idx,time,0,:] = batch_boxes_a[select_idx,time][idy,:]
						boxes_a[select_idx,time,1,:] = batch_boxes_a[select_idx,time][idz,:]
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]==risky)[0]] = 1
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]!=risky)[0]] = 0
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]==agent)[0]] = 1
						boxes_label[select_idx,time,-1] = 1
			return xs_a, boxes_a, boxes_label
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		self.dataset.get_batch()
		iter_per_epoch = len(self.dataset.train_batch)
		#iter_per_epoch = 0
		test_iter_per_epoch = len(self.dataset.test_batch)
		[x, boxes, x_a, boxes_a, boxes_y, y, cross_entropy, accuracy, pred_r] = self.model.model(sess)

		saver = tf.train.Saver(max_to_keep=100)
		if not os.path.isdir(self.ck_dir):
			call('mkdir '+self.ck_dir,shell=True)

		# %% Define loss/eval/training functions
		opt = tf.train.AdamOptimizer(learning_rate=0.00001)
		#optimizer = opt.minimize(cross_entropy)
		grads = opt.compute_gradients(cross_entropy, model.variables)
		optimizer = opt.apply_gradients(grads)

		# %% We now create a new session to actually perform the initialization the
		# variables:
		sess.run(tf.initialize_all_variables())
		#self.model.faster_rcnn.load(self.detector_dir,sess)

		for epoch_i in range(self.n_epochs):
		    batch_order = range(len(self.dataset.train_batch))
		    np.random.shuffle(batch_order)
		    self.dataset.split = 'train'
		    # Use queue to load batch
		    #self.dataset.data_thread = Thread(target=self.dataset.load_data_into_queue)
		    #self.dataset.data_thread.start()
		    loss_epoch = []
		    time_b = time.time()
		    for iter_i in range(0,iter_per_epoch):
			time_a = time.time()
			[batch_xs, batch_xs_a, batch_boxes, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_ys, batch_anno, vid, batch_agent] = self.dataset.get_data_feature(batch_order[iter_i])

			print('Time for loading data:' + str(time.time() - time_a))
			#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
			#time_a = time.time()
			#[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)
			[batch_xs_a_2, batch_boxes_a_2, batch_boxes_ys_2] = select_a()
			print('Time for preprocessing data:' + str(time.time() - time_a))

			time_a = time.time()
			[_, loss] = sess.run([optimizer, cross_entropy], feed_dict={x: batch_xs[:,self.model.start_time:,:], boxes: batch_boxes[:,self.model.start_time:,:], x_a: batch_xs_a_2[:,self.model.start_time:,:], boxes_a: batch_boxes_a_2[:,self.model.start_time:,:], boxes_y:batch_boxes_ys_2[:,self.model.start_time:,:], y: batch_ys[:,self.model.start_time:]})
			loss_epoch.append(loss)
			print('Time for training:' + str(time.time() - time_a))
			print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
			sys.stdout.flush()
		    print('Loss (%d): ' % epoch_i + str(np.mean(loss_epoch)))
		    print('Time for (%d): ' % epoch_i + str(time.time() - time_b))
		    #self.dataset.data_thread.join()

		    if epoch_i % 5 == 0:
			    self.dataset.split = 'test'
			    # Use queue to load batch
			    #self.dataset.data_thread = Thread(target=self.dataset.load_data_into_queue)
			    #self.dataset.data_thread.start()
			    acc = []
			    ap = []
			    for iter_i in range(test_iter_per_epoch):
				[batch_xs, batch_xs_a, batch_boxes, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_ys, batch_anno, vid, batch_agent] = self.dataset.get_data_feature(iter_i)
				[batch_xs_a_2, batch_boxes_a_2, batch_boxes_ys_2] = select_a()
				#[batch_xs, batch_ys, batch_anno] = self.dataset.get_data_q()
				#[batch_rpn, batch_rpn_scales, batch_rpn_info] = self.get_rpn_input(batch_xs)

				[acc_tmp, pred_r_tmp] = sess.run([accuracy, pred_r],feed_dict={x: batch_xs[:,self.model.start_time:,:], boxes: batch_boxes[:,self.model.start_time:,:], x_a: batch_xs_a_2[:,self.model.start_time:,:], boxes_a: batch_boxes_a_2[:,self.model.start_time:,:], boxes_y:batch_boxes_ys_2[:,self.model.start_time:,:], y: batch_ys[:,self.model.start_time:]})
				acc_tmp = np.transpose(acc_tmp, [1,0,2])
				pred_r_tmp = np.transpose(pred_r_tmp, [1,0,2])
				indicator = np.where(np.sum(batch_ys,axis=1)>0)[0]
				acc_tmp = acc_tmp[indicator,:,:]
				pred_r_tmp = pred_r_tmp[indicator,:,:]
				boxes_gt = batch_boxes_ys_2[indicator,self.model.start_time:,:]
				acc_tmp.shape = -1, 1
				pred_r_tmp.shape = pred_r_tmp.shape[0], -1
				boxes_gt.shape = boxes_gt.shape[0], -1
				acc += list(acc_tmp)
				ap_tmp = [average_precision_score(boxes_gt[iii,:],pred_r_tmp[iii,:]) for iii in range(pred_r_tmp.shape[0])]
				ap += ap_tmp
				print iter_i, "Acc:", np.mean(acc_tmp)
				print iter_i, "mAp:", np.mean(ap_tmp)
			    print('Accuracy (%d): ' % epoch_i + str(np.mean(acc)))
			    print('Mean Average Precision (%d): ' % epoch_i + str(np.mean(ap)))
			    sys.stdout.flush()
			    saver.save(sess, os.path.join(self.ck_dir, 'model'), global_step=epoch_i)
			    #self.dataset.data_thread.join()

	def test(self):
		global batch_xs_a, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_agent
		def select_a():
			global select_idx
			xs_a = np.zeros([self.model.batch_size,self.model.n_times,2,batch_xs_a[0,0].shape[1]])
			boxes_a = np.zeros([self.model.batch_size,self.model.n_times,2,batch_boxes_a[0,0].shape[1]])
			boxes_label = np.zeros([self.model.batch_size,self.model.n_times,31])
			for select_idx in xrange(self.model.batch_size):
				if batch_ys[select_idx,0] == 1:
					"""
					# Why the parallel version isn't fater?
					p = mp.Pool(8)
                                	xxx = p.map(select_a_time, range(100))
					p.close()
					xs_a[select_idx,:,:,:] = [yyy[0] for yyy in xxx]
					boxes_a[select_idx,:,:,:] = [yyy[1] for yyy in xxx]
					boxes_label[select_idx,:,:] = [yyy[2] for yyy in xxx]
					"""
					for time in xrange(100):
						anno_boxes_id = np.where(batch_boxes_a_ys[select_idx,time] != None)[0]
						idy = random.choice(anno_boxes_id)
						agent= batch_agent[select_idx,time][idy]
						other = np.where(np.array(batch_agent[select_idx,time])!=agent)[0]
						if len(other)!=0:
							idz = random.choice(other)
						else:
							idz = idy
						risky = batch_agent[select_idx,time][idz]
						xs_a[select_idx,time,0,:] = batch_xs_a[select_idx,time][idy,:]
						xs_a[select_idx,time,1,:] = batch_xs_a[select_idx,time][idz,:]
						boxes_a[select_idx,time,0,:] = batch_boxes_a[select_idx,time][idy,:]
						boxes_a[select_idx,time,1,:] = batch_boxes_a[select_idx,time][idz,:]
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]==risky)[0]] = 1
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]!=risky)[0]] = 0
						boxes_label[select_idx,time,np.where(batch_boxes_ys[select_idx,time,:]==agent)[0]] = 1
						boxes_label[select_idx,time,-1] = 1
			return xs_a, boxes_a, boxes_label

		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		self.dataset.get_batch()
		test_iter_per_epoch = len(self.dataset.test_batch)
		[x, boxes, x_a, boxes_a, boxes_y, y, cross_entropy, accuracy, pred_r] = self.model.model(sess)

		saver = tf.train.Saver(max_to_keep=100)

		# %% We now create a new session to actually perform the initialization the
		#sess.run(tf.initialize_all_variables())

		for model in range(0,85,5):
		    saver.restore(sess, os.path.join(self.ck_dir,'model-'+str(model)))
		    self.dataset.split = 'test'
		    acc = []
		    ap = []
		    for iter_i in range(test_iter_per_epoch):
			[batch_xs, batch_xs_a, batch_boxes, batch_boxes_a, batch_boxes_ys, batch_boxes_a_ys, batch_ys, batch_anno, vid, batch_agent] = self.dataset.get_data_feature(iter_i)
			[batch_xs_a_2, batch_boxes_a_2, batch_boxes_ys_2] = select_a()

			[acc_tmp, pred_r_tmp] = sess.run([accuracy, pred_r],feed_dict={x: batch_xs[:,self.model.start_time:,:], boxes: batch_boxes[:,self.model.start_time:,:], x_a: batch_xs_a_2[:,self.model.start_time:,:], boxes_a: batch_boxes_a_2[:,self.model.start_time:,:], boxes_y:batch_boxes_ys_2[:,self.model.start_time:,:], y: batch_ys[:,self.model.start_time:]})
			acc_tmp = np.transpose(acc_tmp, [1,0,2])
			pred_r_tmp = np.transpose(pred_r_tmp, [1,0,2])
			indicator = np.where(np.sum(batch_ys,axis=1)>0)[0]
			acc_tmp = acc_tmp[indicator,:,:]
			pred_r_tmp = pred_r_tmp[indicator,:,:]
			boxes_gt = batch_boxes_ys_2[indicator,self.model.start_time:,:]
			acc_tmp.shape = -1, 1
			pred_r_tmp.shape = pred_r_tmp.shape[0], -1
			boxes_gt.shape = boxes_gt.shape[0], -1
			acc += list(acc_tmp)
			ap_tmp = [average_precision_score(boxes_gt[iii,:],pred_r_tmp[iii,:]) for iii in range(pred_r_tmp.shape[0])]
			ap += ap_tmp
			#print iter_i, "Acc:", np.mean(acc_tmp)
			print iter_i, "mAp:", np.mean(ap_tmp)
		    print('Accuracy (%d): ' % model + str(np.mean(acc)))
		    print('Mean Average Precision (%d): ' % model + str(np.mean(ap)))
		    sys.stdout.flush()

if __name__ == "__main__":
	sys.path.insert(0, 'data/dashcam/')
	from dashcam import dashcam
	ds = dashcam("label", "Videos", 10, main_dir='data/dashcam/', workers = 1)
	model = ant_1dim_loc(ds, 10, 100,'dashcam_model.npy',start_time=50, device = [0,1,2,3])
	train = trainer(model, ds, 20, 'dashcam_model.npy', device = [0,1,2,3])
	with tf.device('/gpu:'+str(0)):
		train.train()
		#train.test()
		#train.extract_feature(True)
		#train.extract_feature()
		#ds.feat_preprocess()
