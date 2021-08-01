#%%

import numpy as np
import os
dataset_path = "/content/drive/MyDrive/VOCdevkit/VOC2007/"

img_dir = "JPEGImages"
annot_dir = "Annotations"

os.chdir(dataset_path)



obj_class = ['person', # Person
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', # Animal
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', # Vehicle
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor' # Indoor
           ]

import cv2
import xml.etree.ElementTree as Et
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt

from utils import * 

from tqdm import tqdm



#%%
def VOCDataset_encoder( img_file_list, object_class, mode = 'train'):
    '''
    Input:
        img_file_list : list of image file names
        object_class : list of class name of objects 

    Output:
        resized image,  encoded matrix 
    '''

    img_dir = "JPEGImages"
    annot_dir = "Annotations"
    S = 7
    B = 2
    C =  len(object_class)

    label_operater = LabelBinarizer()
    label_operater.fit(object_class)
    
    image_list = []
    labels = []

    if mode =='train':
        for img_file_name in tqdm(img_file_list):
            
            image = cv2.imread(os.path.join(img_dir,img_file_name))
            annot_filename =  img_file_name.split('.')[0]+'.xml'
            
            xml =  open(os.path.join(annot_dir, annot_filename), "r")
            tree = Et.parse(xml)
            root = tree.getroot()
            objects = root.findall("object")
            image_annotation =[]
            for _object in objects:
                name = _object.find('name').text
                bndbox = _object.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                xmax = int(bndbox.find('xmax').text)
                ymin = int(bndbox.find('ymin').text)
                ymax = int(bndbox.find('ymax').text)
                image_annotation.append([name,(xmin+xmax)/2, (ymin+ymax)/2,xmax- xmin, ymax- ymin ])
            
        
            label_matrix = np.zeros((S, S, C + 5 ))
            for box in image_annotation:
                x_normed = 7*box[1]/image.shape[1]
                grid_x = np.int(np.trunc(x_normed))
                x_normed = x_normed - grid_x
                
                y_normed = 7*box[2]/image.shape[0]
                
                grid_y = np.int(np.trunc(y_normed))
                y_normed = y_normed - grid_y
                w_normed = box[4]/image.shape[1]
                h_normed = box[3]/image.shape[0]
                label_matrix[grid_y, grid_x, :5] =  1, x_normed, y_normed , w_normed, h_normed
                label_matrix[grid_y, grid_x, 5:]  = label_operater.transform(np.expand_dims(box[0],axis = 0))[0]
            resized = cv2.resize(image, (448,448), interpolation = cv2.INTER_AREA)
            labels.append(label_matrix)
            image_list.append(resized)
        image_list = np.array(image_list)
        labels = np.array(labels)
        return image_list, labels
    
    else:
        for img_file_name in tqdm(img_file_list):
            image = cv2.imread(os.path.join(img_dir,img_file_name))
            resized = cv2.resize(image, (448,448), interpolation = cv2.INTER_AREA)
            image_list.append(resized)
        image_list = np.array(image_list)

        return image_list

#%%

class Yolo(tf.keras.Model):
    '''
        Yolo model 
    '''

    def __init__(self, S, B,classes_number, **kwargs):

        super(Yolo, self).__init__(name = 'Yolo', **kwargs)

        self.classes_number = classes_number
        self._S = S
        self._B = B

        architecture = [['Conv' ,7, 64, 2], ['Pool', 2,2]] +\
               [['Conv', 3, 192,1], ['Pool', 2,2]] +\
               [['Conv', 1, 128,1],['Conv', 3,256,1],['Conv', 1, 256,1],['Conv', 3, 512,1], ['Pool', 2,2]] +\
               [['Conv', 1, 256,1],['Conv', 3,512,1]]*4 + [['Conv', 1, 512,1],['Conv', 3,1024,1],['Pool', 2,2]]+ \
               [['Conv', 1, 512,1],['Conv', 3,1024,1]]*2 + [['Conv', 3,1024,1],['Conv', 3,1024,2]] +\
               [['Conv', 3,1024,2]]*2 
        self.nn = []
        for i in range(len(architecture)):
            if architecture[i][0] == 'Conv':
                self.nn.append(layers.Conv2D(architecture[i][2], (architecture[i][1], architecture[i][1]),
                    strides =architecture[i][3],  activation = LeakyReLU(0.1), input_shape = (448,448, 3 ), padding = 'same'))
                                    
            if architecture[i][0] == 'Pool':
                self.nn.append(layers.MaxPooling2D(pool_size = (architecture[i][1],architecture[i][1]), 
                                                   strides = architecture[i][2]))

        self.nn.append(layers.Flatten())
        self.nn.append(layers.Dense(4096, activation= LeakyReLU(0.1)))
        self.nn.append(layers.Dropout(0.5))
        self.nn.append(layers.Dense(self._S*self._S*(5*self._B + classes_number),
                                    activation="softmax", 
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                    bias_regularizer=tf.keras.regularizers.l2(0.0005) ) )

    def call(self,inputs):
        x = self.nn[0](inputs)
        for layer in self.nn[1:]:
            x= layer(x)
        
        
        return x

#%%
train_ratio = 0.9

num = len(os.listdir(img_dir))
train_idx = np.random.choice(np.arange(num),round(num*train_ratio) )
test_idx = np.setdiff1d(np.arange(num), train_idx)

train_img_file = np.array(os.listdir(img_dir))[train_idx]
test_img_file =np.array(os.listdir(img_dir))[test_idx]



train_datasets, train_labels = VOCDataset_encoder(train_img_file, object_class= obj_class, mode = 'train')
test_datasets = VOCDataset_encoder(test_img_file, object_class= obj_class, mode = 'test')

np.save("/content/drive/MyDrive/train_datasets", train_datasets)
np.save("/content/drive/MyDrive/train_labels", train_labels)
np.save("/content/drive/MyDrive/test_datasets", test_datasets) 

#%%

train_datasets = np.load("/content/drive/MyDrive/train_datasets.npy")
train_labels = np.load("/content/drive/MyDrive/train_labels.npy")
test_datasets = np.load("/content/drive/MyDrive/test_datasets.npy")

train_for_tf = tf.data.Dataset.from_tensor_slices((train_datasets, train_labels))
train_for_tf = train_for_tf.shuffle(buffer_size = len(train_for_tf)).batch(64)

del(train_datasets)
del(train_labels)
del(test_datasets)


#%%

from tensorflow.keras.losses import Loss


class Yolo_loss(Loss):
    '''
     Loss function of Yolo v1
    '''
    def __init__(self,num_classes = 20 , num_cell = 7, num_boxes = 2, lambda_coord = 0.5, lambda_noobj = 5.0 ):
        super().__init__()
        self._C = num_classes
        self._S = num_cell
        self._B = num_boxes
        self._lambda_coord = lambda_coord
        self._lambda_noobj = lambda_noobj
    def call(self,y_true ,y_pred):
        pred_mat = tf.reshape(y_pred, [-1,self._S,self._S, self._C+self._B*5])

        predicted_classes = pred_mat[:,:,:,(5*self._B):]
        predicted_conf = pred_mat[:,:,:,0:self._B]
        predicted_coord = tf.reshape(pred_mat[:,:,:,self._B:(self._B*5)], [-1,self._S,self._S,self._B,4])
        responsible = tf.cast(tf.reshape(y_true[:,:,:,0], [-1, self._S, self._S, 1]), tf.float32)

        coord_target = tf.reshape(y_true[:,:,:,1:5], [-1,self._S,self._S,1,4])

        coord_target = tf.cast(tf.tile(coord_target, [1, 1, 1, self._B, 1]), tf.float32)

        classes_target = tf.cast(y_true[:,:,:,5:],tf.float32)
        predicted_coord = tf.stack([predicted_coord[:,:,:,:,:2], tf.math.sqrt(predicted_coord[:,:,:,:,2:4])], axis = 5)
        predicted_coord = tf.reshape(predicted_coord, [-1,self._S,self._S,self._B,4])

        iou_predict_truth  = self.compute_iou(predicted_coord, coord_target)

        responsible_idx = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
        responsible_idx = tf.cast((iou_predict_truth >= responsible_idx), tf.float32) * tf.cast(responsible, tf.float32)
        ## responsible box가 두개인 경우 모두 사용하기 위함

        coord_loss = self.compute_coord_loss(coord_target, predicted_coord, responsible_idx)
        res_conf_loss, nores_conf_loss = self.compute_conf_loss(predicted_conf, iou_predict_truth, responsible_idx)
        class_loss = self.compute_class_loss(classes_target, predicted_classes, responsible)

        total_loss = coord_loss+ res_conf_loss + nores_conf_loss + class_loss
        
        return total_loss

    def compute_coord_loss(self,true, pred ,responsible):
        coord_mask = tf.expand_dims(responsible, 4)
        boxes_delta = coord_mask * tf.square(true - pred)
        coord_loss_return = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') *self._lambda_coord
        return coord_loss_return

    def compute_conf_loss(self,pred, iou, responsible):
        not_responsible = tf.ones_like(responsible, dtype=tf.float32) - responsible

        responsible_delta = responsible * (iou - pred)
        responsible_loss = tf.reduce_mean(tf.reduce_sum(tf.square(responsible_delta), axis=[1, 2, 3]),  name='respons_conf_loss')

        not_responsible_delta = not_responsible * pred
        not_responsible_loss = tf.reduce_mean(tf.reduce_sum(tf.square(not_responsible_delta), axis=[1, 2, 3]), name='norespons_conf_loss')*self._lambda_noobj

        return responsible_loss , not_responsible_loss

    def compute_class_loss(self,true, pred, detector):
        class_loss_return = detector*(true-pred)
        class_loss_return = tf.reduce_mean(tf.reduce_sum(tf.square(class_loss_return), axis = [1,2,3]), name = 'class_loss')
        return class_loss_return
    def compute_iou(self, boxes1, boxes2):
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0],axis =4)
        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0] ,axis = 4)
        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                  (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                  (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])
        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

#%%

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
    if step <75:
        return self.initial_learning_rate 
    elif step <105:
        return self.initial_learning_rate /10
    else:
        return self.initial_learning_rate /100


optimizer = tf.keras.optimizers.SGD(learning_rate= MyLRSchedule(0.001), momentum= 0.9)
loss_metric = tf.keras.metrics.Mean()


#%%

epochs = 135

model = Yolo(classes_number= len(obj_class) , S =7 , B = 2)

yolo_loss = Yolo_loss()

for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    print("Epoch: %d" % epoch)
    for step, (x_batch_train,y_batch_train) in enumerate(train_for_tf):
        with tf.GradientTape(persistent=True) as tape:
            result = model(tf.cast(x_batch_train, tf.float32))
            # Compute reconstruction loss
            loss = yolo_loss(y_batch_train,result )

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_metric(loss)
        if step % 10 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

tf.saved_model.save(model, '/content/drive/MyDrive/data/Yolo_v1')

#%%
model = tf.saved_model.load('/content/drive/MyDrive/data/Yolo_v1')


#%%

def test_time_dectection(model, image, object_class):
    '''
    Input:
        model : Yolo model
        image : image
        object_class : list of object names
    Output:
        plotting estimated object box 
    '''

    resize = cv2.resize(image, (448,448),interpolation = cv2.INTER_AREA)
    resize = tf.cast(resize, tf.float32)
    resize = tf.reshape(resize, [1,448,448,3])
    result = model(resize)

    offset = tf.reshape(tf.cast([[i%7, i//7] for i in np.arange(49)],tf.float32), [7,7,2])

    pred_mat = tf.reshape(result, [ 7,7,30])

    ### box 재조정
    pred_mat[:,:,2:4] = pred_mat[:,:,2:4] + offset
    pred_mat[:,:,6:8] = pred_mat[:,:,6:8] + offset

    pred_mat[:,:,2] =  pred_mat[:,:,2]/7 *image.shape[1]
    pred_mat[:,:,3] =  pred_mat[:,:,3]/7 *image.shape[0]
    pred_mat[:,:,4] = pred_mat[:,:,4] *image.shape[1]
    pred_mat[:,:,5] = pred_mat[:,:,5] *image.shape[0]

    pred_mat[:,:,6] =  pred_mat[:,:,6]/7 *image.shape[1]
    pred_mat[:,:,7] =  pred_mat[:,:,7]/7 *image.shape[0]
    pred_mat[:,:,8] = pred_mat[:,:,8] *image.shape[1]
    pred_mat[:,:,9] = pred_mat[:,:,9] *image.shape[0]

    predicted_classes = pred_mat[:,:,10:]
    predicted_classes = tf.reshape(predicted_classes , [49,20])

    predicted_classes = tf.reshape(tf.tile(predicted_classes, [1,2]), [98,20])

    predicted_conf = pred_mat[:,:,:2]
    predicted_conf = tf.reshape(predicted_conf, [98,1])

    predicted_coord = tf.reshape(pred_mat[:,:,2:10], [98,4])

    result_box = tf.concat([predicted_conf, predicted_coord, predicted_classes], axis = 1)

    result_box = np.asarray(result_box)

    NMS_idx = filtering_and_non_max_suppression(result_box, object_class)

    for k, class_name in NMS_idx:
        image_rec = cv2.rectangle(image,(np.int(result_box[1] -result_box[3]/2),np.int(result_box[2] -result_box[4]/2)),
                                (np.int(result_box[1] +result_box[3]/2),np.int(result_box[2] +result_box[4]/2)),(255,0,0), 2)
        cv2.putText(image_rec, class_name.astype(str),(result_box[1], result_box[2]), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

    plt.imshow(image)
    plt.show()

#%% Detection


image = image_read(img_dir, annot_dir, 1)

test_time_dectection(image,model,obj_class)