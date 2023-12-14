import re

import keras
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Add
from keras.layers import Conv2DTranspose
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.regularizers import l2

import tensorflow as tf
from tensorflow import keras

import cv2
import matplotlib.pyplot as plt
import numpy as np

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

DEF_FILEPATH = './checkpoint/ORANGES-weights.hdf5'
FILEPATH = 'Scripts/checkpoint_new/model-weights.hdf5'  # Direccion donde se va a almacenar el modelo entrenado
EPOCHS = 100
BATCH_SIZE = 16


def get_MN_layers(layer_C, prevlayer_M=None, layer_name=None, feature_size=256, last_layer=False):
    layerC_reduced = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1,
                                         padding='same',
                                         name='{}_lateral_conn_own'.format(layer_name))(layer_C)
    if last_layer:
        conv_layer_conn_M = layerC_reduced
    else:
        name_Mlayer = re.sub('[^A-Za-z0-9]+', '', prevlayer_M.name)
        layer_M_upsampled = keras.layers.Conv2DTranspose(filters=256, kernel_size=2, strides=2,
                                                         name='{}_upsampled_own'.format(name_Mlayer))(prevlayer_M)
        layer_conn_M = keras.layers.Concatenate(axis=3, name='{}_concatenate_own'.format(layer_name))(
            [layer_M_upsampled, layerC_reduced])
        conv_layer_conn_M = keras.layers.Conv2D(256, (1, 1), padding='same',
                                                name='{}_conv_concatenate_own'.format(layer_name))(layer_conn_M)

    return conv_layer_conn_M


def get_PN_layers(layer_M, layer_name=None, feature_size=256):
    layer_PConvApplied = keras.layers.Conv2D(feature_size, (3, 3), padding='same',
                                             name='{}_conv_own'.format(layer_name))(
        layer_M)
    layer_P_bn = keras.layers.BatchNormalization(name='{}_bn_own'.format(layer_name))(layer_PConvApplied)
    layer_P = keras.layers.Activation('relu', name="{}_relu_own".format(layer_name))(layer_P_bn)

    return layer_P


def SqueezeExcitation_block(input_tensor, ratio=16, name_own="def_own"):
    channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling2D(name=name_own)(input_tensor)

    # Paso de Excitation (calibración importancia)
    excitation = Dense(channels // ratio, activation='relu')(squeeze)
    excitation_drop = Dropout(0.45)(excitation)
    excitation = Dense(channels, activation='sigmoid')(excitation_drop)
    scaled_input = Multiply()([input_tensor, excitation])
    return scaled_input


def weighted_SqueezeExcitation(P2, P3, P4, P5):
    P2_se = SqueezeExcitation_block(P2, name_own="GAP_P2_own")
    P3_se = SqueezeExcitation_block(P3, name_own="GAP_P3_own")
    P4_se = SqueezeExcitation_block(P4, name_own="GAP_P4_own")
    P5_se = SqueezeExcitation_block(P5, name_own="GAP_P5_own")

    upsampled_map3 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(P3_se)
    upsampled_map4 = Conv2DTranspose(256, (4, 4), strides=(4, 4), padding='same')(P4_se)
    upsampled_map5 = Conv2DTranspose(256, (8, 8), strides=(8, 8), padding='same')(P5_se)

    concatenated_maps = tf.keras.layers.concatenate([P2_se, upsampled_map3, upsampled_map4, upsampled_map5], axis=-1)

    conv_concatenated = keras.layers.Conv2D(256, (1, 1), padding='same', name="conv_concatenated_own")(
        concatenated_maps)

    weighted_layer = GlobalAveragePooling2D(name="weighted_layer_own")(conv_concatenated)

    add_output_bn = keras.layers.BatchNormalization(name="add_output_bn_own")(weighted_layer)

    Xdense = keras.layers.Dense(64, activation='relu', name="weighted_64_dense_own", kernel_regularizer=l2(0.01))(
        add_output_bn)
    X_out = keras.layers.BatchNormalization()(Xdense)
    return X_out


def weighted_sum_B1(P2, P3, P4, P5, layers_qty=3):
    feature2 = GlobalAveragePooling2D(name="GAP_P2_own")(P2)
    feature3 = GlobalAveragePooling2D(name="GAP_P3_own")(P3)
    feature4 = GlobalAveragePooling2D(name="GAP_P4_own")(P4)
    feature5 = GlobalAveragePooling2D(name="GAP_P5_own")(P5)

    if layers_qty == 3:
        gap_outputs = [feature3, feature4, feature5]
        first_drop = 0.3
    else:
        gap_outputs = [feature2, feature3, feature4, feature5]
        first_drop = 0.5

    concatenate_outputs = tf.keras.layers.Concatenate()(gap_outputs)
    concatenate_outputs_bn = keras.layers.BatchNormalization()(concatenate_outputs)
    dp_all_concat = Dropout(first_drop)(concatenate_outputs_bn)

    weights_layer = tf.keras.layers.Dense(layers_qty, activation="softmax", name='weights_layer_own')(dp_all_concat)
    weights_reshaped = tf.reshape(weights_layer, (-1, layers_qty, 1))

    pooled_outputs_expanded = [tf.expand_dims(output, axis=1) for output in gap_outputs]
    pooled_outputs_concat = tf.keras.layers.Concatenate(axis=1)(pooled_outputs_expanded)

    weighted_out = tf.multiply(weights_reshaped, pooled_outputs_concat)
    x = tf.reduce_sum(weighted_out, axis=1)
    Xbn = keras.layers.BatchNormalization()(x)

    if layers_qty == 3:
        Xdense = keras.layers.Dense(128, activation='relu', name="weighted_128_dense_own")(Xbn)
        Xdense_drop = Dropout(0.2)(Xdense)
        X_pre_out = keras.layers.Dense(64, activation='relu', name="weighted_64_dense_own")(Xdense_drop)
        X_out = X_pre_out
    else:
        Xdense = keras.layers.Dense(128, activation='relu', name="weighted_128_dense_own")(Xbn)
        Xdense_drop = Dropout(0.2)(Xdense)
        X_pre_out = keras.layers.Dense(64, activation='relu', name="weighted_64_dense_own")(Xdense_drop)
        X_out = X_pre_out

    return X_out


def feature_combination_linear(P2, P3, P4, P5):
    dense_units = 32
    dropout_rate = 0.2

    #################################################################
    feature0 = GlobalAveragePooling2D(name="GAP_P2_own")(P2)
    bn_feature0 = keras.layers.BatchNormalization()(feature0)
    dp0 = Dropout(dropout_rate)(bn_feature0)  # 0.15
    preds0 = Dense(dense_units, activation='relu', name="Dense_P2_own")(dp0)
    preds0 = keras.layers.BatchNormalization()(preds0)

    #################################################################
    feature1 = GlobalAveragePooling2D(name="GAP_P3_own")(P3)
    bn_feature1 = keras.layers.BatchNormalization()(feature1)
    dp1 = Dropout(dropout_rate)(bn_feature1)  # 0.15
    preds1 = Dense(dense_units, activation='relu', name="Dense_P3_own")(dp1)
    preds1 = keras.layers.BatchNormalization()(preds1)

    #################################################################
    feature2 = GlobalAveragePooling2D(name="GAP_P4_own")(P4)
    bn_feature2 = keras.layers.BatchNormalization()(feature2)
    dp2 = Dropout(dropout_rate)(bn_feature2)  # 0.2
    preds2 = Dense(dense_units, activation='relu', name="Dense_P4_own")(dp2)
    preds2 = keras.layers.BatchNormalization()(preds2)

    #################################################################
    feature3 = GlobalAveragePooling2D(name="GAP_P5_own")(P5)
    bn_feature3 = keras.layers.BatchNormalization()(feature3)
    dp3 = Dropout(dropout_rate)(bn_feature3)  # 0.2
    preds3 = Dense(dense_units, activation='relu', name="Dense_P5_own")(dp3)
    preds3 = keras.layers.BatchNormalization()(preds3)

    #################################################################
    # Concatenacion de los mapas de cada nivel de la piramide
    concat = keras.layers.Concatenate(axis=1)([preds0, preds1, preds2, preds3])
    last_dense = keras.layers.Dense(64, activation='relu', name="last_dense_own", kernel_regularizer=l2(0.01))(concat)
    bn_last_dense = keras.layers.BatchNormalization()(last_dense)

    return bn_last_dense


def feature_combination_linear_B1(P3, P4, P5):
    dense_units = 32
    dropout_rate = 0.25

    #################################################################
    feature1 = GlobalAveragePooling2D(name="GAP_P3_own")(P3)
    bn_feature1 = keras.layers.BatchNormalization()(feature1)
    dp1 = Dropout(dropout_rate)(bn_feature1)  # 0.15
    preds1 = Dense(dense_units, activation='relu', name="Dense_P3_own")(dp1)
    preds1 = keras.layers.BatchNormalization()(preds1)

    #################################################################
    feature2 = GlobalAveragePooling2D(name="GAP_P4_own")(P4)
    bn_feature2 = keras.layers.BatchNormalization()(feature2)
    dp2 = Dropout(dropout_rate)(bn_feature2)  # 0.2
    preds2 = Dense(dense_units, activation='relu', name="Dense_P4_own")(dp2)
    preds2 = keras.layers.BatchNormalization()(preds2)

    #################################################################
    feature3 = GlobalAveragePooling2D(name="GAP_P5_own")(P5)
    bn_feature3 = keras.layers.BatchNormalization()(feature3)
    dp3 = Dropout(dropout_rate)(bn_feature3)  # 0.2
    preds3 = Dense(dense_units, activation='relu', name="Dense_P5_own")(dp3)
    preds3 = keras.layers.BatchNormalization()(preds3)

    #################################################################
    concat = keras.layers.Concatenate(axis=1)([preds1, preds2, preds3])
    last_dense = keras.layers.Dense(64, activation='relu', name="last_dense_own", kernel_regularizer=l2(0.01))(concat)
    bn_last_dense = keras.layers.BatchNormalization()(last_dense)

    return bn_last_dense


def recalibration_module(layer):
    print("recalibration_module")
    Xavg = tf.keras.layers.GlobalAveragePooling2D()(layer)
    Xmax = tf.keras.layers.GlobalMaxPooling2D()(layer)
    Xintegrate = tf.keras.layers.Add()([Xavg, Xmax])
    Xintegrate = tf.keras.layers.Activation("sigmoid")(Xintegrate)
    layer_recalibrated = tf.keras.layers.Multiply()([Xintegrate, layer])

    return layer_recalibrated


def define_data_augmentation(image=None, graph=False):
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(factor=(-0.15, 0.15), fill_mode="constant"),
        keras.layers.GaussianNoise(stddev=0.1, seed=25)
    ], name='data_augmentation_own')

    if graph:
        image = tf.expand_dims(image, 0)
        data_aug_applied = data_augmentation(image)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(tf.squeeze(image / 255.0))

        plt.subplot(1, 2, 2)
        plt.title("Augmented")
        plt.imshow(tf.squeeze(data_aug_applied))
        plt.show()

    return data_augmentation


def set_trainable(base_model, n=0, prefix=list()):
    if prefix:
        for layer in base_model.layers:
            if layer.name.startswith(tuple(prefix)):
                layer.trainable = True
    else:
        for layer in base_model.layers[-n:]:
            layer.trainable = True

    return base_model


def configurations_FPN(IMAGES_HEIGHT, IMAGES_WIDTH, model_type="B0", partition=None):
    global FILEPATH
    try:
        from Scripts.Read_FileImage import RESNET_FLAG, window_size, MODEL_DEF
    except:
        from Read_FileImage import RESNET_FLAG, window_size, MODEL_DEF

    if RESNET_FLAG:
        try:
            from Scripts.FPN_development import base_model_fpn
        except:
            from FPN_development import base_model_fpn

        if partition or partition >= 0:
            FILEPATH = 'Modelos_Finales/Resnet/{}/normal_eq/Resnet_weighted_sum_4l.hdf5'.format(partition)
        else:
            FILEPATH = 'Modelos_Finales/Resnet/normal_eq/Resnet_weighted_sum_4l.hdf5'
        base_model, output_model = base_model_fpn(IMAGES_HEIGHT, IMAGES_WIDTH)
    else:
        try:
            from Scripts.FPN_EfficientNet_development import base_model_fpn, base_efficientnet
        except:
            from FPN_EfficientNet_development import base_model_fpn, base_efficientnet
        if partition or partition >= 0:
            FILEPATH = 'Modelos_Finales/{}/{}/B0_train_{}.hdf5'.format(MODEL_DEF, partition, window_size)
        else:
            FILEPATH = 'Modelos_Finales/{}/B0_train.hdf5'.format(MODEL_DEF)

        if "original" in model_type:
            print("MODELO ORIGINAL")
            base_model, output_model = base_efficientnet(IMAGES_HEIGHT, IMAGES_WIDTH, base_model=model_type)
        else:
            base_model, output_model = base_model_fpn(IMAGES_HEIGHT, IMAGES_WIDTH, base_model=model_type)

    new_model = keras.models.Model(inputs=base_model.input, outputs=output_model)
    return new_model


def train_FPN(model, train_data, dir_train_index=""):
    try:
        from Scripts.utils_prediction import save_model_history
    except:
        from utils_prediction import save_model_history

    model_history = None
    steps_per_epoch = round(train_data[0].shape[0] / BATCH_SIZE)
    n_updates = EPOCHS * steps_per_epoch
    lr_schedule_cosine = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3,
                                                                   decay_steps=n_updates,
                                                                   alpha=1e-5,
                                                                   name=None)
    opt_adam = keras.optimizers.Adam(learning_rate=lr_schedule_cosine)

    model.compile(optimizer=opt_adam,
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])

    checkpoint = ModelCheckpoint(FILEPATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    callbacks_list = [checkpoint]

    model_history = model.fit(train_data[0], train_data[1],
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_split=0.2,
                              shuffle=True,
                              callbacks=callbacks_list,
                              verbose=0)
    model.load_weights(FILEPATH)

    try:
        from Scripts.Read_FileImage import ubuntu, MODEL_DEF, window_size
    except:
        from Read_FileImage import ubuntu, MODEL_DEF, window_size

    save_model_history(model_history, dir_train_index)
    return model, model_history
