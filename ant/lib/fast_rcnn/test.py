from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
import cPickle
import heapq
#from utils.blob import im_list_to_blob
import os, pdb
import math
from rpn_msr.generate import imdb_proposals_det
import tensorflow as tf
#from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import matplotlib.pyplot as plt

def bbox_transform_inv(boxes, deltas):
    #if boxes.shape[0] == 0:
    #    return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = tf.cast(boxes, tf.float32)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = []
    dy = []
    dw = []
    dh = []
    for idx in xrange(21):
	dx.append(deltas[:, idx*4])
	dy.append(deltas[:, idx*4+1])
	dw.append(deltas[:, idx*4+2])
	dh.append(deltas[:, idx*4+3])
    #dx = deltas[:, 0::4]
    #dy = deltas[:, 1::4]
    #dw = deltas[:, 2::4]
    #dh = deltas[:, 3::4]
    dx = tf.pack(dx)
    dy = tf.pack(dy)
    dw = tf.pack(dw)
    dh = tf.pack(dh)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.exp(dw) * widths
    pred_h = tf.exp(dh) * heights
    """
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    """
    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h
    pred_boxes = []
    for idx in xrange(21):
	pred_boxes.append(x1[:,idx])
	pred_boxes.append(y1[:,idx])
	pred_boxes.append(x2[:,idx])
	pred_boxes.append(y2[:,idx])
    pred_boxes = tf.transpose(tf.pack(pred_boxes))

    return pred_boxes

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = tf.reduce_max(tf.pack([tf.shape(im) for im in ims]),reduction_indices=0)
    num_images = len(ims)
    #blob = tf.zeros((num_images, max_shape[0], max_shape[1], 3))
    blob = []
    for i in xrange(num_images):
        im = ims[i]
	im = tf.concat(0,[im,tf.zeros((max_shape[0]-tf.shape(im)[0],tf.shape(im)[1],3))])
	im = tf.concat(1,[im,tf.zeros((tf.shape(im)[0],max_shape[1]-tf.shape(im)[1],3))])
	blob.append(im)
        #blob[i, 0:tf.shape(im)[0], 0:tf.shape(im)[1], :] += im

    return tf.pack(blob)

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    #im_orig = im.astype(np.float32, copy=True)
    im_orig = tf.cast(im, tf.float32)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = tf.shape(im_orig)
    im_size_min = tf.cast(tf.reduce_max(im_shape[0:2]),tf.float32)
    im_size_max = tf.cast(tf.reduce_min(im_shape[0:2]),tf.float32)

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / tf.cast(im_size_min, tf.float32)
        # Prevent the biggest axis from being more than MAX_SIZE
        check=tf.cast(tf.greater(tf.round(im_scale * im_size_max), tf.cast(tf.constant(cfg.TEST.MAX_SIZE),tf.float32)),tf.float32)
        im_scale = check * float(cfg.TEST.MAX_SIZE) / tf.cast(im_size_max, tf.float32) + (1-check) * float(target_size) / tf.cast(im_size_min, tf.float32)
        im = tf.image.resize_images(im_orig, tf.cast(tf.cast(im_shape[0],tf.float32)*im_scale,tf.int32), tf.cast(tf.cast(im_shape[1],tf.float32)*im_scale,tf.int32),method=0)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, tf.pack(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.TEST.HAS_RPN:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
    else:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    """
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for idx in xrange(21):
	x1.append(tf.maximum(boxes[:, idx*4],0))
	y1.append(tf.maximum(boxes[:, idx*4+1],0))
	x2.append(tf.minimum(boxes[:, idx*4+2],tf.cast(im_shape[1] - 1,tf.float32)))
	y2.append(tf.minimum(boxes[:, idx*4+3],tf.cast(im_shape[0] - 1,tf.float32)))
    #x1 = tf.maximum(boxes[:, 0::4], 0)
    #y1 = tf.maximum(boxes[:, 1::4], 0)
    #x2 = tf.minimum(boxes[:, 2::4], tf.cast(im_shape[1] - 1,tf.float32))
    #y2 = tf.minimum(boxes[:, 3::4], tf.cast(im_shape[0] - 1,tf.float32))
    x1 = tf.pack(x1)
    y1 = tf.pack(y1)
    x2 = tf.pack(x2)
    y2 = tf.pack(y2)
    boxes = []
    for idx in xrange(21):
	boxes.append(x1[:,idx])
	boxes.append(y1[:,idx])
	boxes.append(x2[:,idx])
	boxes.append(y2[:,idx])
    boxes = tf.transpose(tf.pack(boxes))
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, boxes=None):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = tf.expand_dims(tf.pack([tf.cast(tf.shape(im_blob)[1],tf.float32), tf.cast(tf.shape(im_blob)[2],tf.float32), im_scales[0]]),0)
    # forward pass
    if cfg.TEST.HAS_RPN:
        #feed_dict={net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
	net.data = blobs['data']
	net.im_info = blobs['im_info']
	net.keep_prob = 1.0
    else:
        #feed_dict={net.data: blobs['data'], net.rois: blobs['rois'], net.keep_prob: 1.0}
	net.data = blobs['data']
	net.im_info = blobs['rois']
	net.keep_prob = 1.0

    #cls_score, cls_prob, bbox_pred, rois, pool_5 = sess.run([net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'),net.get_output('rois'),net.get_output('pool_5')], feed_dict=feed_dict)
    cls_score = tf.concat(0,[tf.zeros((1,21)),net.get_output('cls_score')])
    cls_prob = tf.concat(0,[tf.zeros((1,21)),net.get_output('cls_prob')])
    bbox_pred = tf.concat(0,[tf.zeros((1,84)),net.get_output('bbox_pred')])
    rois = tf.concat(0,[tf.zeros((1,5)),net.get_output('rois')])
    pool_5 = tf.concat(0,[tf.zeros((1,7,7,512)),net.get_output('pool_5')])
    
    if cfg.TEST.HAS_RPN:
        #assert tf.shape(im_scales)[0] == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]


    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, tf.shape(im))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = tf.tile(boxes, (1, tf.shape(scores)[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
	pool_5 = pool_5[inv_index, :, :, :]

    return scores, pred_boxes, pool_5


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt 
    #im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4] 
        score = dets[i, -1] 
        if score > thresh:
            #plt.cla()
            #plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.gca().text(bbox[0], bbox[1] - 2,
                 '{:s} {:.3f}'.format(class_name, score),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))
    #plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds,:]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, imdb, weights_filename , max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        if vis:
            image = im[:, :, (2, 1, 0)] 
            plt.cla()
            plt.imshow(image)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(image, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets
        if vis:
           plt.show()
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

