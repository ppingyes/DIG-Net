from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
from six.moves import xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('QUALITY', '10',"""the quality of image""")
tf.app.flags.DEFINE_string('MODEL', 'DIG',"""the name of model""")




path ="./data/train_jp_"+FLAGS.QUALITY+".tfrecords"


def read_and_decode(filename):


		IMAGE_SIZE = 32
		filename_queue = tf.train.string_input_producer([filename])

		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_example,
				 features={
						 'label': tf.FixedLenFeature([], tf.string),
						 'guide' : tf.FixedLenFeature([], tf.string),
						 'depth' : tf.FixedLenFeature([], tf.string)
				 })
		label = tf.decode_raw(features['label'], tf.uint8)
		label = tf.reshape(label, [IMAGE_SIZE, IMAGE_SIZE,1])
		label = tf.cast(label, tf.float32) 

		img = tf.decode_raw(features['guide'], tf.uint8)
		img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE,1])
		img = tf.cast(img, tf.float32) 

		dep = tf.decode_raw(features['depth'], tf.uint8)
		dep = tf.reshape(dep, [IMAGE_SIZE, IMAGE_SIZE,1])
		dep = tf.cast(dep, tf.float32) 

		return label,img,dep


def DIG_net():  

	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.001) 
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.0001, shape=shape)
		return tf.Variable(initial)

	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def get_sobel(img):
			sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
			sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
			sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

			image = tf.placeholder(tf.float32, shape=[None, None])

			filtered_x = tf.nn.conv2d(img, sobel_x_filter,strides=[1, 1, 1, 1], padding='SAME')
			filtered_y = tf.nn.conv2d(img, sobel_y_filter,strides=[1, 1, 1, 1], padding='SAME')
			s_img = tf.abs(filtered_x)+tf.abs(filtered_y)
			return filtered_x,filtered_y,s_img

	def rmse(or_img,dis_img):

		l = tf.pow((or_img-dis_img),2)
		loss = tf.sqrt(tf.reduce_mean(l))

		return loss


	
	BATCH_SIZE = 64

	label,img,dep = read_and_decode(path)
	cnn_label,cnn_image,cnn_depth = tf.train.shuffle_batch([label,img,dep],batch_size=BATCH_SIZE, 
																			capacity=200000,
																			min_after_dequeue=100)
	#tf.summary.image("cnn_label", cnn_label)
	#tf.summary.image("cnn_image", cnn_image)
	#tf.summary.image("cnn_depth", cnn_depth)


	pool1 = tf.nn.avg_pool(cnn_image, ksize=[1,9,9,1],strides = [1,1,1,1],padding = 'SAME')

	diffcolor = pool1 - cnn_image

	#tf.summary.image("diffcolor", diffcolor)

	#cov1
	w_conv11 = weight_variable([5, 5, 1, 32])
	b_conv11 = bias_variable([32])
	conv11 = conv2d(diffcolor, w_conv11) + b_conv11
	h_conv11 = tf.nn.relu(conv11)

	#cov2 
	w_conv12 = weight_variable([5, 5, 32, 32])
	b_conv12 = bias_variable([32])
	conv12 = conv2d(h_conv11, w_conv12) + b_conv12
	h_conv12 = tf.nn.relu(conv12)

	
	#second layer

	pool2 = tf.nn.avg_pool(cnn_depth, ksize=[1,9,9,1],strides = [1,1,1,1],padding = 'SAME')

	diffdepth = pool2 - cnn_depth

	#tf.summary.image("diffdepth", diffdepth)

	#cov1
	w_conv21 = weight_variable([5, 5, 1, 32])
	b_conv21 = bias_variable([32])
	conv21 = conv2d(diffdepth, w_conv21) + b_conv21
	h_conv21 = tf.nn.relu(conv21)

	#cov2 
	w_conv22 = weight_variable([5, 5, 32, 32])
	b_conv22 = bias_variable([32])
	conv22 = conv2d(h_conv21, w_conv22) + b_conv22
	h_conv22 = tf.nn.relu(conv22)


	#third layer
	w_conv31 = weight_variable([5, 5, 1, 32])      
	b_conv31 = bias_variable([32])
	conv31   = conv2d(cnn_depth, w_conv31) + b_conv31
	h_conv31 = tf.nn.relu(conv31)

	w_conv32 = weight_variable([5, 5, 32, 32])      
	b_conv32 = bias_variable([32])
	conv32   = conv2d(h_conv31, w_conv32) + b_conv32
	concat32 = tf.concat([conv32 ,h_conv11,h_conv21],3)
	h_conv32 = tf.nn.relu(concat32)

	w_conv33 = weight_variable([5, 5, 96, 32])      
	b_conv33 = bias_variable([32])
	conv33   = conv2d(h_conv32, w_conv33) + b_conv33
	concat33 = tf.concat([conv33 ,h_conv12,h_conv22],3)
	h_conv33 = tf.nn.relu(concat33)

	w_conv34 = weight_variable([5, 5, 96,32])      
	b_conv34 = bias_variable([32])
	conv34   = conv2d(h_conv33, w_conv34) + b_conv34
	h_conv34 = tf.nn.relu(conv34)

	w_conv35 = weight_variable([5, 5,32, 1])      
	b_conv35 = bias_variable([1])
	conv35   = conv2d(h_conv34, w_conv35) + b_conv35
	y_conv   = tf.nn.relu(conv35)

	_,_,re_sobel = get_sobel(y_conv)
	_,_,or_sobel = get_sobel(cnn_label)

	#tf.summary.image("or_sobel", or_sobel)
	#tf.summary.image("re_sobel", re_sobel)

	#tf.summary.image("out", y_conv)


	loss_img   = rmse(cnn_label,y_conv)
	loss_sobel = rmse(or_sobel,re_sobel)

	loss = loss_img+ 0.1*loss_sobel

	#tf.summary.scalar("loss_img", loss_img)
	#tf.summary.scalar("loss_sobel", loss_sobel)
	#tf.summary.scalar("loss", loss)


	optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)


	TRAIN_TIMES = 300000
	

	saver = tf.train.Saver(tf.all_variables(),max_to_keep = 200000)
	config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
		

	with tf.Session(config = config) as sess:

		tf.initialize_all_variables().run()
		threads = tf.train.start_queue_runners(sess=sess)
		#summary_writer = tf.summary.FileWriter(tempSavePath, sess.graph.as_graph_def())
		#merged = tf.summary.merge_all()
		#train_writer = tf.summary.FileWriter('./tmp/' + FLAGS.MODEL+'/'+FLAGS.QUALITY+'/',sess.graph)
	 
		spath = './ckpt/' + FLAGS.MODEL+'/'+FLAGS.QUALITY+'/'
		if not os.path.exists(spath):
				os.makedirs(spath)
		print('Initialized!')


		mean_loss = 0
		mean_loss_sobel = 0
		mean_loss_img = 0
		cc = 1
		for step in xrange(TRAIN_TIMES):

			loss_,loss_sobel_,loss_img_,_ = sess.run([loss,loss_sobel,loss_img,optimizer])    
			mean_loss += loss_
			mean_loss_sobel += loss_sobel_
			mean_loss_img += loss_img_			

			cc += 1
			if step % 100 == 0:
				
				print(' loss: %.5f,loss_sobel: %.5f,loss_img: %.5f,step:%d' % (mean_loss/cc,mean_loss_sobel/cc,mean_loss_img/cc,step))
				mean_loss = 0
				mean_loss_sobel = 0
				mean_loss_img = 0
				cc = 1
			if step % 1000 == 0 or (step + 1) == TRAIN_TIMES:
				print(FLAGS.MODEL+'_'+FLAGS.QUALITY,"_step:",step,"-----save")
				checkpoint_path = os.path.join(spath,'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)

				#summary = sess.run(merged)
				#train_writer.add_summary(summary, step)


if __name__ ==  '__main__':

	DIG_net()