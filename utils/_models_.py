from __future__ import print_function
import os,sys,pdb # pdb.set_trace() # !import code; code.interact(local=vars())
from _helper_basics_ import *
from _helper_DNN_ import *

## Hashing
def conv12_layer(
    inputs,
    activation='relu',
        batch_normalization=True,
        conv_first=True):
    conv = Conv2D(  12,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4),
                        )
    pool = MaxPooling2D(pool_size=(2, 4), 
                        strides=(2, 4), 
                        padding='valid')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return pool(x)
def conv34_layer(
    inputs,
    activation='relu',
        batch_normalization=True,
        conv_first=True):
    conv = Conv2D(  6,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def conv5_layer(
    inputs,
    activation='relu',
        batch_normalization=True,
        conv_first=True):
    conv = Conv2D(  1,
                    kernel_size=(1,9),
                    strides=1,
                    padding='valid',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))
    x = inputs
    # x = Reshape((8,40))(inputs)
    # x = Permute((2,1))(x)
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def f_model(input_shape):
    # v3 but reduce embedding from 2106 to 256
    # 1) additional conv layer
    # 2) additional dense layer
    ## 
    inputs = Input(input_shape)
    ## 
    x = conv12_layer(inputs=inputs)
    x = conv12_layer(inputs=x)
    x = conv34_layer(inputs=x)
    x = conv34_layer(inputs=x)
    x = conv5_layer(inputs=x)
    ## 
    outputs = Flatten()(x)
    return Model(inputs=inputs, outputs=outputs)


## Training
def f_siamese(mdl_in, input_shape):
    Inp_left = Input(shape=input_shape)
    Inp_rite = Input(shape=input_shape)
    # 
    out_left=mdl_in(Inp_left) 
    out_rite=mdl_in(Inp_rite) 
    y = concatenate([out_left,out_rite])
    # 
    y = Dense(432, activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(y)
    y = Dropout(0.2)(y)
    y = Dense(32, activation='relu', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(y)
    y = Dropout(0.2)(y)
    outputs = Dense(1, activation='sigmoid', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(y)
    ## Instantiate model.
    return Model([Inp_left,Inp_rite], outputs)
