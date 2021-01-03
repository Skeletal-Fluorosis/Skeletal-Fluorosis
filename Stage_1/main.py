
#stage1

import configparser
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import regularizers, BatchNormalization, advanced_activations, Conv2DTranspose, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from help_functions import *
from extract_patches import *
from typing import Any, Union
import os, sys

#============ Model ============
def DUnet(n_ch,patch_height,patch_width):
    inputs = Input(shape=(n_ch,patch_height,patch_width))

    conv1 = CONV(inputs, 8)
    dens1_48 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', data_format='channels_first')(conv1)
    dens1_24 = Conv2D(4, (1, 1), strides=(2, 2), padding='same', data_format='channels_first')(conv1)
    dens1_12 = Conv2D(4, (1, 1), strides=(4, 4), padding='same', data_format='channels_first')(conv1)
    dens1_6 = Conv2D(4, (1, 1), strides=(8, 8), padding='same', data_format='channels_first')(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = CONV(pool1, 16)
    dens2_48 = Conv2DTranspose(4, (1, 1), strides=(2, 2), padding='same', data_format='channels_first')(conv2)
    dens2_24 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', data_format='channels_first')(conv2)
    dens2_12 = Conv2D(4, (1, 1), strides=(2, 2), padding='same', data_format='channels_first')(conv2)
    dens2_6 = Conv2D(4, (1, 1), strides=(4, 4), padding='same', data_format='channels_first')(conv2)

    pool2 = MaxPooling2D((2, 2))(conv2)
    conc1 = concatenate([pool2, dens1_12], axis=1)
    conv3 = CONV(conc1, 32)
    dens3_48 = Conv2DTranspose(4, (1, 1), strides=(4, 4), padding='same', data_format='channels_first')(conv3)
    dens3_24 = Conv2DTranspose(4, (1, 1), strides=(2, 2), padding='same', data_format='channels_first')(conv3)
    dens3_12 = Conv2D(4, (1, 1), strides=(1, 1), padding='same', data_format='channels_first')(conv3)

    pool3 = MaxPooling2D((2, 2))(conv3)
    conc2 = concatenate([pool3, dens1_6, dens2_6], axis=1)
    conv4 = CONV(conc2, 64)
    dens4_48 = Conv2DTranspose(4, (1, 1), strides=(8, 8), padding='same', data_format='channels_first')(conv4)
    dens4_24 = Conv2DTranspose(4, (1, 1), strides=(4, 4), padding='same', data_format='channels_first')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    conc3 = concatenate([up1, dens1_12, dens2_12, dens3_12], axis=1)
    conv5 = CONV(conc3, 32)
    dens5_48 = Conv2DTranspose(4, (1, 1), strides=(4, 4), padding='same', data_format='channels_first')(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    conc4 = concatenate([up2, dens1_24, dens2_24, dens3_24, dens4_24], axis=1)
    conv6 = CONV(conc4, 16)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    conc5 = concatenate([up3, dens1_48, dens2_48, dens3_48, dens4_48, dens5_48], axis=1)
    conv7 = CONV(conc5, 8)

    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv7)
    conv8 = core.Reshape((2,patch_height*patch_width))(conv8)
    conv8 = core.Permute((2,1))(conv8)
    conv8 = core.Activation('softmax')(conv8)
    model = Model(inputs=inputs, outputs=conv8)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    return model
def CONV(inputs, n_filter):
    conv = Conv2D(n_filter, (5, 5), padding='same', data_format='channels_first')(inputs)
    conv = BatchNormalization()(conv)
    conv = advanced_activations.PReLU()(conv)
    conv = Conv2D(n_filter, (5, 5), padding='same', data_format='channels_first')(conv)
    return conv
def step_decay(epoch):
    lrate = 0.01
    if epoch > 5:
        lrate = 0.005
    if epoch > 10:
        lrate = 0.002
    if epoch > 15:
        lrate = 0.001
    if epoch > 20:
        lrate = 0.0005
    return lrate

config = configparser.RawConfigParser()
config.read('configuration.txt')
path_data = config.get('data paths', 'path_local')
name_experiment = config.get('experiment name', 'name')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV')
)

N_sample = min(patches_imgs_train.shape[0],40)
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]


#============ Train ============
model = DUnet(n_ch, patch_height, patch_width)
json_string = model.to_json()
open('./'+name_experiment+'/'+name_experiment +'_architecture.json', 'w').write(json_string)

checkpointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5',verbose=1,monitor='val_loss',mode='auto',save_best_only=True)
patches_masks_train = masks_Unet(patches_masks_train)
model.fit(patches_imgs_train, patches_masks_train,nb_epoch=N_epochs,batch_size=batch_size,verbose=2,shuffle=True,validation_split=0.1,callbacks=[checkpointer])
lrate = LearningRateScheduler(step_decay)

model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
score = model.evaluate(patches_imgs_test, masks_Unet(patches_masks_test), verbose=0)



#============ Test ============
config = configparser.RawConfigParser()
config.read('configuration.txt')
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup') 

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

result_dir = name_experiment
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

if nohup:
    print ("\n2. Run the prediction on GPU  with nohup")
    os.system(run_GPU +' nohup python -u ./predict.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.nohup')
else:
    print ("\n2. Run the prediction on GPU (no nohup)")
    os.system(run_GPU +' python ./predict.py')
