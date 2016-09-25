import numpy as np
import json, os, pdb, cv2, sys
from subprocess import call
from multiprocessing import Pool

def read_imgs(name):
	return cv2.resize(cv2.imread(name),(640,360))

class dashcam(object):
	def __init__(self, label_dir, video_dir, batch_size, frame_cnt = 100, H = 360, W = 640, C = 3, workers = 8, main_dir = '.'):
		self.main_dir = main_dir
		self.data_list = np.load(main_dir+'/list.npy').tolist()
		self.label_dir = main_dir+'/'+label_dir
		self.video_dir = main_dir+'/'+video_dir
		self.frames_dir = main_dir+'/frames'
		self.batch_dir = main_dir+'/batch'
		self.batch_size = batch_size
		self.frame_cnt = frame_cnt
		self.H = H
		self.W = W
		self.C = C
		self.workers = workers
		self.train_batch = None
		self.test_batch = None
		self.label = None
		self.n_classes = 2

	def get_label(self):
		if os.path.isfile(self.main_dir+'/'+'labels.json'):
			self.label = json.load(open(self.main_dir+'/'+'labels.json'))
		else:
			label = self.get_label_from_txt()
			json.dump(label, open(self.main_dir+'/'+'labels.json','w'))
			self.label = label

	def get_label_from_txt(self):
		label = {}
		labels = os.listdir(self.label_dir)
		for ele in labels:
			tmp = open(self.label_dir+'/'+ele).read().split('\n')[:-1]
			vid = ele.split('.')[0]
			if vid not in label.keys():
				label[vid] = {}
			for line in tmp:
				data = line.split(' ')
				obj_id = int(data[0])
				if obj_id not in label[vid].keys():
					label[vid][obj_id] = {}
					label[vid][obj_id]['bbox'] = []
					label[vid][obj_id]['frame'] = []
					label[vid][obj_id]['outside'] = []
					label[vid][obj_id]['occluded'] = []
					label[vid][obj_id]['label'] = data[9].split('"')[1]
					label[vid][obj_id]['accident'] = []
				label[vid][obj_id]['bbox'].append([int(bbox_info) for bbox_info in data[1:5]])
				label[vid][obj_id]['frame'].append(int(data[6]))
				label[vid][obj_id]['outside'].append(int(data[7]))
				label[vid][obj_id]['occluded'].append(int(data[8]))
				if len(data) > 10:
					label[vid][obj_id]['accident'].append(True)
				else:
					label[vid][obj_id]['accident'].append(False)
		return label

	def read_imgs(self, name):
		return cv2.resize(cv2.imread(name),(self.W,self.H))

	def get_batch(self):
		if not os.path.isdir(self.batch_dir):
			call('mkdir '+self.batch_dir,shell=True)
		elif len(os.listdir(self.batch_dir+'/train')) > 0:
			self.train_batch = [self.batch_dir+'/train/'+ele for ele in os.listdir(self.batch_dir+'/train')]
			self.test_batch = [self.batch_dir+'/test/'+ele for ele in os.listdir(self.batch_dir+'/test')]
			return
		if not os.path.isdir(self.frames_dir+'/positive/') or not os.path.isdir(self.frames_dir+'/negative/'):
			self.get_frame()
		if not os.path.isdir(self.batch_dir+'/train') or len(os.listdir(self.batch_dir+'/train')) == 0:
			self.generate_batch('train')
		self.train_batch = [self.batch_dir+'/train/'+ele for ele in os.listdir(self.batch_dir+'/train')]
		if not os.path.isdir(self.batch_dir+'/test') or len(os.listdir(self.batch_dir+'/test')) == 0:
			self.generate_batch('test')
		self.test_batch = [self.batch_dir+'/test/'+ele for ele in os.listdir(self.batch_dir+'/test')]

	def generate_batch(self, split):
		path = self.batch_dir+'/'+split
		if not os.path.isdir(path):
			call('mkdir '+path,shell=True)
		pos = [self.frames_dir+'/positive/'+ele for ele in self.data_list[split]['positive']]
		neg = [self.frames_dir+'/negative/'+ele for ele in self.data_list[split]['negative']]
		total = pos + neg
		np.random.shuffle(total)
		batch_num = len(total) / self.batch_size
		for idx in xrange(batch_num):
			name = [ele.split('/')[-1] for ele in total[idx*self.batch_size:(idx+1)*self.batch_size]]
			pos_neg = [ele.split('/')[-2] for ele in total[idx*self.batch_size:(idx+1)*self.batch_size]]
			data_dir = total[idx*self.batch_size:(idx+1)*self.batch_size]
			imgs = np.zeros((self.batch_size,self.frame_cnt,self.H,self.W,self.C), dtype=np.uint8)
			for idy, ele in enumerate(data_dir):
				img_names = [ele+'/{:03}.jpg'.format(idz+1) for idz in range(self.frame_cnt)]
				p = Pool(self.workers)
				imgs[idy,:,:,:,:] = p.map(read_imgs, img_names)
				p.close()
			np.savez(path+'/batch{:03}.npz'.format(idx+1), imgs=imgs, vid=name, pos_neg=pos_neg)
			sys.stdout.write(str(int(((idx+1)/float(batch_num))*100))+'%\r')
                	sys.stdout.flush()
			
	def get_frame(self):
		if not os.path.isdir(self.frames_dir):
			call('mkdir '+self.frames_dir,shell=True)
		positive_dir = self.frames_dir+'/positive'
		if not os.path.isdir(positive_dir):
			call('mkdir '+positive_dir,shell=True)
		negative_dir = self.frames_dir+'/negative'
		if not os.path.isdir(negative_dir):
			call('mkdir '+negative_dir,shell=True)
		pos_video = [self.video_dir+'/positive/'+ele for ele in os.listdir(self.video_dir+'/positive')]
		for ele in pos_video:
			frames_dir = positive_dir+'/'+ele.split('/')[-1].split('.')[0]
			if not os.path.isdir(frames_dir):
				call('mkdir '+frames_dir,shell=True)
			call("ffmpeg -i "+ele+" -qscale:v 1 -f image2 "+frames_dir+"/%3d.jpg",shell=True)
		neg_video = [self.video_dir+'/negative/'+ele for ele in os.listdir(self.video_dir+'/negative')]
		for ele in neg_video:
			frames_dir = negative_dir+'/'+ele.split('/')[-1].split('.')[0]
			if not os.path.isdir(frames_dir):
				call('mkdir '+frames_dir,shell=True)
			call("ffmpeg -i "+ele+" -qscale:v 1 -f image2 "+frames_dir+"/%3d.jpg",shell=True)

	def get_data(self, split, idx):
		if self.train_batch == None or self.test_batch == None:
			self.get_batch()
		if self.label == None:
			self.get_label()
		if split == 'train':
			batch = np.load(self.train_batch[idx])
		elif split == 'test':
			batch = np.load(self.test_batch[idx])
			print self.test_batch[idx]
		imgs = batch['imgs']
		label = []
		annotate = []
		for idx, vid in enumerate(batch['vid']):
			if batch['pos_neg'][idx] == 'positive':
				for k in self.label[vid].keys():
					if sum(self.label[vid][k]['accident']) > 0:
						idy = np.where(np.array(self.label[vid][k]['accident'])==True)[0][0]
						label.append([0]*idy + [1]*(self.frame_cnt-idy))
						break
				annotate.append(self.label[vid])
			else:
				label.append([0]*self.frame_cnt)
				annotate.append({})
		return np.array(imgs, dtype = np.float32), np.array(label, dtype = np.int32), annotate

if __name__ == "__main__":
	ds = dashcam("label", "Videos", 10)
	[img, label, annotate] = ds.get_data('test',42)
	pdb.set_trace()
