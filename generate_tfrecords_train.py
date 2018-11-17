import os
from PIL import Image
import numpy as np
import tensorflow as tf





def get_patch(label_img, color_img, depth_img, stride, size_image):


  sub_patch_label = []
  sub_patch_depth = []
  sub_patch_color = []

  [hei,wid] = label_img.shape

  count = 0
  for x in range(0,hei-size_image,stride):
    for y in range(0,wid-size_image,stride):

     
      tmp_label = label_img[x:x+size_image, y:y+size_image]
      tmp_color = color_img[x:x+size_image, y:y+size_image]
      tmp_depth = depth_img[x:x+size_image, y:y+size_image]
      
      
      sub_patch_label.append(tmp_label)
      sub_patch_color.append(tmp_color)
      sub_patch_depth.append(tmp_depth)

  return sub_patch_label, sub_patch_color, sub_patch_depth



def create_record():
    c = 0

    image_size = 32
    stride = 20

    label_path = './data/train/depth/'
    color_path = './data/train/color/'

  
    for quality in [10,20,30,40]:
        depth_file ='./data/jpeg/train/'+str(quality)+'/'
        
        writer = tf.python_io.TFRecordWriter("./data/train_jp_"+str(quality)+".tfrecords")
       
       
        count = 0
        
        for ln in range(1,106):
            print('image num:%d'%ln)
  
            label_img_path = label_path + str(ln) +'.png'
            color_img_path = color_path + str(ln) +'.png'
            depth_img_path = depth_file + 'depth/'+ str(ln) +'.jpg'
           

            label_img = np.array(Image.open(label_img_path))
            color_img = np.array(Image.open(color_img_path).convert('L') )
            depth_img = np.array(Image.open(depth_img_path))

           
            sub_label,sub_gray,sub_depth = get_patch(label_img, color_img, depth_img, stride, image_size) 
            
            l = len(sub_gray)
            c = l+c
            for i in range(0,l):

                sub_la_byte = sub_label[i].tobytes()
                sub_co_byte = sub_gray[i].tobytes()
                sub_de_byte = sub_depth[i].tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sub_la_byte])),
                    'guide': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sub_co_byte])),
                    'depth': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sub_de_byte]))
                }))
        
                writer.write(example.SerializeToString())
    writer.close()
    print("length:",c)

if __name__ == '__main__':
    create_record()