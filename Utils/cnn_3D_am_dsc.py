# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:33:01 2020

@author: lhy
"""

from keras.models import Model, Sequential
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from custom_layers import SeparableConvolution3D
from keras.layers import (
    Input,
    Activation,
    merge,#tensorflow.keras与keras 版本不一致，tf里的低
    Dense,
    Flatten,
    Dropout,
    Multiply,
    Lambda,
    concatenate,
    BatchNormalization
)

from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras import backend as K
from keras import regularizers


def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

#自定义注意力模块，相关参考Github
def ssa_3D(input_sa):
    avg_x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_sa)
    assert avg_x.shape[-1] == 1
    
    max_x = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_sa)
    assert max_x.shape[-1] == 1
    
    concat = concatenate([avg_x, max_x])
    ssa_refined = Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1),padding='same',
                         activation='hard_sigmoid', kernel_initializer='he_normal')(concat)
    
    return Multiply()([input_sa, ssa_refined])

# 组合模型
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        print('original input shape:', input_shape)
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        print('original input shape:', input_shape)
        # orignal input shape: 1,7,7,200

        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])
        print('change input shape:', input_shape)

        # 用keras中函数式模型API，不用序贯模型API
        # 构建模型
        input = Input(shape=input_shape)# 张量流输入
        conv1 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 2),
                       kernel_regularizer=regularizers.l2(0.01),data_format = 'channels_last')(input)
        #bn1 = BatchNormalization(axis=4, epsilon=1.001e-5)(conv1)
        act1 = Activation('relu')(conv1)
        pool1 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(act1)#padding=same时，尺寸与kernelsize无关
        am1 = ssa_3D(pool1)  #注意力模块
        
        conv2 =SeparableConvolution3D(filters=192, kernel_size=(3, 3, 3), strides=(1, 1, 2),
                       kernel_regularizer=regularizers.l2(0.01),data_format = 'channels_last')(am1)
        #bn2 = BatchNormalization(axis=4, epsilon=1.001e-5)(conv2)
        act2 = Activation('relu')(conv2)
        drop1 = Dropout(0.5)(act2)
        pool2 = AveragePooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(drop1)
        
        conv3 = SeparableConvolution3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                       kernel_regularizer=regularizers.l2(0.01),data_format = 'channels_last')(pool2)
        #bn3 = BatchNormalization(axis=4, epsilon=1.001e-5)(conv3)
        act3 = Activation('relu')(conv3)
        drop2 = Dropout(0.5)(act3)
                
        flatten1 = GlobalAveragePooling3D()(drop2)
        #flatten1 = Flatten()(attention_mul)
        fc1 = Dense(200, kernel_regularizer=regularizers.l2(0.01))(flatten1)
        act3 = Activation('relu')(fc1)

        
        # 输入分类器
        # Classifier block
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(act3)

        model = Model(inputs=input, outputs=dense)
        
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        # (1,7,7,200),16
        return ResnetBuilder.build(input_shape, num_outputs)

def main():
    model = ResnetBuilder.build_resnet_8((1, 7, 7, 15), 15)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()

if __name__ == '__main__':
    main()
