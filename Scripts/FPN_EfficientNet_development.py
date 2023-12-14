import numpy as np
import tensorflow as tf
import keras

from keras.layers import Dropout, Input
from keras.initializers import HeNormal, GlorotNormal
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1

try:
    from Scripts.utils import (feature_combination_linear, feature_combination_linear_B1,
                               recalibration_module,
                               weighted_SqueezeExcitation,
                               weighted_sum_B1,
                               deep_semantic_embedding_combined,
                               deep_semantic_embedding, define_data_augmentation, get_MN_layers, get_PN_layers,
                               set_trainable)
except:
    from utils import (feature_combination_linear, feature_combination_linear_B1,
                       recalibration_module,
                       weighted_SqueezeExcitation,
                       weighted_sum_B1,
                       deep_semantic_embedding_combined,
                       deep_semantic_embedding, define_data_augmentation, get_MN_layers, get_PN_layers, set_trainable,)

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def base_efficientnet(IMAGES_HEIGHT, IMAGES_WIDTH, CHANNELS=3, base_model="B0"):
    input_shape = (IMAGES_HEIGHT, IMAGES_WIDTH, CHANNELS)
    input_tensor = Input(shape=input_shape)

    data_augmentation = define_data_augmentation()
    x = data_augmentation(input_tensor)

    if "_B0" in base_model:
        print("MODELO B0", base_model)
        base_model_obj = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
        n = 16  # capa 7
    elif "_B1" in base_model:
        base_model_obj = EfficientNetB1(weights="imagenet", include_top=False, input_tensor=x)
        n = 105  # capa 6 y 7

    base_model_obj.trainable = False
    base_model_obj = set_trainable(base_model_obj, n=n)
    last_output = base_model_obj.output

    Xavg = keras.layers.GlobalAveragePooling2D(name="GAP_own")(last_output)
    Xbn_avg = keras.layers.BatchNormalization()(Xavg)
    Xdp_avg = Dropout(0.4)(Xbn_avg)
    Xdense = keras.layers.Dense(64, activation='relu', name="dense_own")(Xdp_avg)
    Xbn = keras.layers.BatchNormalization()(Xdense)
    Xdp = Dropout(0.4)(Xbn)
    Xfinal = keras.layers.Dense(1, activation="sigmoid", name="final_own", kernel_initializer=GlorotNormal())(Xdp)

    return base_model_obj, Xfinal


def base_model_fpn(IMAGES_HEIGHT, IMAGES_WIDTH, CHANNELS=3, base_model="_B0", file_name=None):
    input_shape = (IMAGES_HEIGHT, IMAGES_WIDTH, CHANNELS)
    input_tensor = Input(shape=input_shape)

    data_augmentation = define_data_augmentation()
    x = data_augmentation(input_tensor)

    if "_B0" in base_model:
        print("MODELO B0", base_model)
        base_model_obj = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
        layer_names = ["block2b_add",
                       "block3b_add",
                       "block5c_add",
                       "top_activation"]
        n = 16  # Solo ultima capa
    elif "_B1" in base_model:
        print("MODELO B1", base_model)
        base_model_obj = EfficientNetB1(weights="imagenet", include_top=False, input_tensor=x)
        layer_names = ["block2c_add",
                       "block3c_add",
                       "block5d_add",
                       "top_activation"]
        n = 105  # capa 6 y 7

    base_model_obj.trainable = False
    base_model_obj = set_trainable(base_model_obj, n=n)

    layer_outputs = [base_model_obj.get_layer(name).output for name in layer_names]
    C2, C3, C4, C5 = layer_outputs

    # Creacion de nivel P5
    M5 = get_MN_layers(C5, layer_name="M5", last_layer=True)
    M5 = recalibration_module(M5)
    P5 = get_PN_layers(M5, layer_name="P5")

    # Creacion de nivel P4
    M4 = get_MN_layers(C4, prevlayer_M=M5, layer_name="M4")
    M4 = recalibration_module(M4)
    P4 = get_PN_layers(M4, layer_name="P4")

    # Creacion de nivel P3
    M3 = get_MN_layers(C3, prevlayer_M=M4, layer_name="M3")
    M3 = recalibration_module(M3)
    P3 = get_PN_layers(M3, layer_name="P3")

    # Creacion de nivel P2
    M2 = get_MN_layers(C2, prevlayer_M=M3, layer_name="M2")
    M2 = recalibration_module(M2)
    P2 = get_PN_layers(M2, layer_name="P2")

    if "_B1" in base_model:
        all_feat = weighted_sum_B1(P2, P3, P4, P5, layers_qty=3)
        # all_feat = feature_combination_linear_B1(P2, P3, P4, P5)
    else:
        # Modulo Combinacion Ponderada
        all_feat = weighted_SqueezeExcitation(P2, P3, P4, P5)

        # Modulo Combinacion Lineal
        # all_feat = feature_combination_linear(P2, P3, P4, P5)

    out = keras.layers.Dense(1, activation="sigmoid", name="Output_own", kernel_initializer=GlorotNormal())(all_feat)
    return base_model_obj, out
