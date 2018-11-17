from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np

import tensorflow as tf
from PIL import Image
import math



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('IMAGE_NUM', '1',
                           """the Number of image""")

tf.app.flags.DEFINE_string('TESTING_QUALITY', '10',
                           """the quality of testing""")



def DIG_net():  
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.001) 
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.zeros(shape) 
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def psnr(img1, img2):
        img1 = np.array(img1,dtype=np.float16)
        img2 = np.array(img2,dtype=np.float16)

        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    
    cnn_depth = tf.placeholder(tf.float32,shape=None)
    cnn_color = tf.placeholder(tf.float32,shape=None)
   

    pool1 = tf.nn.avg_pool(cnn_color, ksize=[1,9,9,1],strides = [1,1,1,1],padding = 'SAME')

    diffcolor = pool1 - cnn_color

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
    concat32 = tf.concat([conv33 ,h_conv12,h_conv22],3)
    h_conv33 = tf.nn.relu(concat32)

    w_conv34 = weight_variable([5, 5, 96,32])      
    b_conv34 = bias_variable([32])
    conv34   = conv2d(h_conv33, w_conv34) + b_conv34
    h_conv34 = tf.nn.relu(conv34)

    w_conv35 = weight_variable([5, 5,32, 1])      
    b_conv35 = bias_variable([1])
    conv35   = conv2d(h_conv34, w_conv35) + b_conv35
    y_conv   = tf.nn.relu(conv35)
  

    y_conv = tf.minimum(y_conv,255)
    y_conv = tf.maximum(y_conv,0)


  

    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess = tf.InteractiveSession(config = config)
    
    saver = tf.train.Saver(tf.all_variables())
    ckptFile = './ckpt/'+FLAGS.TESTING_QUALITY+'/model.ckpt-0'
    saver.restore(sess,ckptFile)


    labelpath = './data/test/depth/'+FLAGS.IMAGE_NUM+'.png'
    imagepath = './data/test/color/'+FLAGS.IMAGE_NUM+'.png'
    depthpath = './data/test/jpeg/'+FLAGS.TESTING_QUALITY+'/'+FLAGS.IMAGE_NUM+'.jpg'


    label   = np.array(Image.open(labelpath),dtype= np.float32)
    image   = np.array(Image.open(imagepath).convert(mode='L'),dtype= np.float32)
    jpg     = np.array(Image.open(depthpath),dtype= np.float32)
    
    [hei,wid] = label.shape

    #label   = np.reshape(label,[1,hei,wid,1])
    color_img   = np.reshape(image,[1,hei,wid,1])
    depth_img   = np.reshape(jpg  ,[1,hei,wid,1])

    re_array= sess.run(y_conv,feed_dict={cnn_depth: depth_img, cnn_color:color_img})



    re_image   = Image.fromarray(np.uint8(re_array[0,:,:,0]))
    
    spath = './data/test/re/'+FLAGS.TESTING_QUALITY+'/'
    if not os.path.exists(spath):
      os.makedirs(spath)
    re_image_name = spath+FLAGS.IMAGE_NUM+'_'+FLAGS.TESTING_QUALITY+'.png'
    re_image.save(re_image_name)




    psnr_re = psnr(re_image,label)
    psnr_de = psnr(jpg,label)

    print('No.%s,quality:%s,psnr_re:%-5.5f,psnr_de:%-5.5f'%(FLAGS.IMAGE_NUM,FLAGS.TESTING_QUALITY,psnr_re,psnr_de))

    sess.close()

if __name__ ==  '__main__':

    DIG_net()





