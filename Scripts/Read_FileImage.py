import os
import os.path as path
import csv
import argparse

import pandas as pd
import numpy as np
import cv2
from time import sleep, time

import tensorflow as tf
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

RESNET_FLAG = False

ubuntu = True
filter_flag_csv = False  # restringir las muestras de entrada a un rango de naranjas
mean_real_values = True  # Uso de cantidades medias como ground truth

window_size = 7  # Cantidad de fotogramas correspondientes a un arbol (1, 3, 7, 11)

FLAG_RESIZE = False  # Bandera para aplicar el redimensionamiento
NO_EQUALIZATION = False  # Bandera para aplicar ecualizacion por histograma
CLAHE_EQ = False  # Bandera para aplicar ecualizacion Clahe

if RESNET_FLAG:
    IMG_DEF_HEIGHT, IMG_DEF_WIDTH, CHANNELS_DEF = 480, 384, 3  # Valores utilizados para redimension y en funcion para la construccion de la red neuronal
    MODEL_DEF = "_RESNET50"
else:
    IMG_DEF_HEIGHT, IMG_DEF_WIDTH, CHANNELS_DEF = 224, 224, 3
    MODEL_DEF = "_B0_original"  # Nombre con el que se va a guardar el modelo y todos los archivos referentes a su entrenamiento.
    # Si se coloca la palabra "original" realiza el entrenamiento con la EfficientNet,
    # cualquier otro nombre con la FPN y el tipo de configuración del módulo descomentado.

MAIN_FOLDER = None
MAIN_FOLDER_IMGS = None


def resize_img(frame):
    global IMG_DEF_HEIGHT, IMG_DEF_WIDTH
    if FLAG_RESIZE:
        new_size = (IMG_DEF_WIDTH, IMG_DEF_HEIGHT)
        frame_resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    else:
        frame_resized = frame
    return frame_resized


def equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_yuv)
    # Canales Y: luminancia, cr: crominancia roja, cb:crominancia azul
    if not NO_EQUALIZATION:
        if CLAHE_EQ:
            clip = 3.0  # umbral que limita el contraste
            tile = 8  # tamaño de la division de las celdas
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            y_clahe = clahe.apply(y)
            img_eq = cv2.merge((y_clahe, cr, cb))
        else:
            y_eq = cv2.equalizeHist(y)
            img_eq = cv2.merge((y_eq, cr, cb))
    else:
        img_eq = img_yuv

    if RESNET_FLAG:
        equalized_img = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2BGR)
    else:
        equalized_img = cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2RGB)
    return equalized_img


# Devuelve un diccionario con información extraida de los archivos CSV con el numero de frame, cantidad de naranjas,
# frame central, y el correspondiente fotograma (puede ser redimensionado).
def read_csv_data():
    global MAIN_FOLDER, MAIN_FOLDER_IMGS
    global IMG_DEF_HEIGHT, IMG_DEF_WIDTH
    k = 0
    dict_data = dict()
    for directory in os.listdir(MAIN_FOLDER):
        if os.path.isdir(os.path.join(MAIN_FOLDER, directory)):
            for sub_dir in os.listdir("{}/{}".format(MAIN_FOLDER, directory)):
                error_read = False
                error_count = 0
                dataset = np.array([])
                if sub_dir.split(".")[-1] in ["csv"]:
                    location = "{}/{}/{}".format(MAIN_FOLDER, directory, sub_dir)
                    with open(location) as f:
                        df = pd.read_csv(f, sep=',', header=None)
                        df.columns = df.iloc[0]  # Obtener la primera fila
                        df = df[1:]  # Eliminar la primera fila del DataFrame
                        df = df.astype({"id_frame": "str", "quantity": "int", "img_center": "str"})

                        if window_size != 11:
                            rows_to_select = []
                            selected_rows = df[df['id_frame'] == df['img_center']]

                            size = int(window_size / 2)
                            if size > 5:
                                size = 5

                            for index in selected_rows.index:
                                range_rows = range(index - size, index + size + 1)  # Calcula el rango de filas
                                rows_to_select.extend(range_rows)  # Agrega las filas al conjunto

                            rows = list(set(rows_to_select))
                            df = df.loc[rows]
                            df = df.reset_index(drop=True)
                            df['numero'] = df["id_frame"].str.extract(r'_(\d+)\.png').astype(int)
                            df = df.sort_values(by='numero', ascending=True)
                            df.drop('numero', axis=1, inplace=True)
                            df.reset_index(drop=True, inplace=True)

                        if mean_real_values:
                            df_error_stats = df.groupby('img_center')['quantity'].transform('mean')
                            df_error_stats.rename('mean_quantity', inplace=True)
                            df = pd.concat([df, df_error_stats], axis=1)
                            df['mean_quantity'] = df['mean_quantity'].round(2)
                            df['quantity'] = df['mean_quantity']
                            df.drop('mean_quantity', axis=1, inplace=True)

                        dataset = df.values
                        if filter_flag_csv:
                            dataset_out = dataset[(dataset[:, 1] >= 56) | (dataset[:, 1] < 25)]
                            unique_values_out = np.unique(dataset_out[:, 2])
                            unique_indices_out = np.where(np.isin(dataset[:, 2], unique_values_out, invert=True))[0]
                            dataset = dataset[unique_indices_out]

                        images = np.empty((dataset.shape[0],), dtype=np.ndarray)
                        for i in range(len(dataset)):
                            img_file = "{}/{}/{}".format(MAIN_FOLDER_IMGS, directory, dataset[i][0])
                            if path.exists(img_file):
                                img = cv2.imread(img_file)
                                img = equalization(img)
                                img = resize_img(img)
                            else:
                                img = np.zeros((IMG_DEF_HEIGHT, IMG_DEF_WIDTH, 3), dtype=np.ndarray)
                                error_count += 1

                            if error_count > images.size * 0.1:
                                error_read = True
                                break

                            img_array = np.asarray(img)
                            images[i] = img_array
                        """La función reshape(-1, 1) se utiliza para convertir el array en un array de una sola columna
                        antes de concatenarlo horizontalmente con dataset.
                        La opción -1 permite determinar automáticamente el tamaño de la primera dimensión del nuevo array
                        en función del tamaño del array original."""
                        dataset = np.hstack((dataset, images.reshape(-1, 1)))
                    k += 1
                if dataset.size > 0 and not error_read:
                    dict_data[k] = dataset
    return dict_data


def k_fold(dict_data):
    try:
        from Scripts.utils import configurations_FPN, train_FPN, FILEPATH, \
            define_data_augmentation
        from Scripts.utils_prediction import (get_predictions,
                                              scatter_plot_results, histogram_samples, graph_training_result,
                                              create_files_with_results)
    except:
        from utils import configurations_FPN, train_FPN, FILEPATH, \
            define_data_augmentation
        from utils_prediction import (get_predictions,
                                      scatter_plot_results, histogram_samples, graph_training_result,
                                      create_files_with_results)

    global IMG_DEF_HEIGHT, IMG_DEF_WIDTH
    k = len(dict_data.keys())
    kf = KFold(n_splits=k)

    X = list()
    Y = list()
    MAIN_IMAGES = list()
    OTHER_IMAGES = list()

    results_training = []
    results_test = []

    min_global = 0
    max_global = 0

    for k, val in dict_data.items():
        array_data = val[:, [3, 1]]
        train_array_imagenes = np.zeros(
            (val[:, [3]].shape[0], IMG_DEF_HEIGHT, IMG_DEF_WIDTH, CHANNELS_DEF))  # Arreglo con imagenes leidas
        train_array_labels = val[:, [1]].reshape(val[:, [1]].shape[0], 1)  # Ground truth - naranjas por fotograma
        main_images_labels = val[:, [2]].reshape(val[:, [1]].shape[0], 1)  # Nombres centrales del fotograma
        other_images_labels = val[:, [0]].reshape(val[:, [1]].shape[0], 1)  # Nombres de todos los fotogramas

        for i, matriz in enumerate(array_data[:, 0]):
            train_array_imagenes[i, :, :, :] = matriz

        X.append(train_array_imagenes)
        Y.append(train_array_labels)

        # Obtencion de valores maximox y minimos de todo el conjunto del dataset.
        if min_global == 0 or min(train_array_labels) < min_global:
            min_global = min(train_array_labels)
        if max(train_array_labels) > max_global:
            max_global = max(train_array_labels)

        MAIN_IMAGES.append(main_images_labels)
        OTHER_IMAGES.append(other_images_labels)

    for cont, (train_index, test_index) in enumerate(kf.split(X)):
        # Creacion de la arquitectura del modelo a utilizar
        model_FPN = configurations_FPN(IMG_DEF_HEIGHT, IMG_DEF_WIDTH, model_type=MODEL_DEF, partition=cont)

        # Dividir los datos en entrenamiento y prueba según los índices del split actual
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
        main_images_test = [MAIN_IMAGES[i] for i in test_index]
        other_images_test = [OTHER_IMAGES[i] for i in test_index]

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)

        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        # Normalizacion al valor mas grande del conjunto de entrenamiento actual
        y_max = np.max(y_train)
        y_train = y_train / y_max
        y_test = np.clip(y_test / y_max, 0, 1)

        train_index_str = ''.join(map(str, train_index))
        dir_train_index = "{}_".format(str(cont)) + train_index_str

        # Entrenamiento del modelo
        model_FPN, model_FPN_history = train_FPN(model_FPN, [X_train.astype('float32'), y_train.astype('float32')],
                                                 dir_train_index)

        # Obtencion de resultados con el modelo entrenado, uso de datos del conjunto de entrenamiento y posterior el
        # empleo del dataset de prueba
        for index in train_index:
            result_predictions = get_predictions(X[index].astype('float32'), Y[index].astype('float32') / y_max,
                                                 [MAIN_IMAGES[index]],
                                                 [OTHER_IMAGES[index]],
                                                 model_FPN,
                                                 [index],
                                                 model_dir_file=None,
                                                 # Direccion donde exista un modelo con los pesos ya entreanados (extension valida: hdf5)
                                                 MaxQtyTraining=y_max,
                                                 loop=cont)
            results_training.append(result_predictions)

        result_predictions = get_predictions(X_test.astype('float32'), y_test.astype('float32'),
                                             main_images_test,
                                             other_images_test,
                                             model_FPN,
                                             test_index,
                                             model_dir_file=None,
                                             MaxQtyTraining=y_max,
                                             loop=test_index[0])
        results_test.append(result_predictions)

        #################################### Graficos
        # 1. Scatter Plot: error ground truth - predictions.
        # 2. Histrogramas: frecuencia de cantidades utilizadas para el entrenamiento y la prueba.
        # 3. Curvas de entrenamiento
        str_test_index = [str(i) for i in
                          test_index]  # Nombre con el que va a ser guardado los graficos del dataset de prueba
        scatter_plot_results(results_test[-1], title_img="Test Dataset_Partition_{}".format(''.join(str_test_index)),
                             model_name=MODEL_DEF, max_val_data=max_global)
        histogram_samples(y_train * y_max, y_test * y_max, min_global[0], max_global[0], train_index, test_index)
        graph_training_result(model_FPN_history, title="Test Dataset_Partition_{}".format(''.join(str_test_index)))

        del X_train, X_test, y_train, y_test, main_images_test, other_images_test
        tf.keras.backend.clear_session()
        del model_FPN

    create_files_with_results(results_training, results_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def_csv = "Direccion archivos CSV con cantidades de naranjas por fotograma"
    def_img = "Direccion de fotogramas de las distintas vias"

    parser.add_argument("--folder_csv", type=str, help="Ubicacion de las carpetas con los archivos csv",
                        default=def_csv)
    parser.add_argument("--folder_images", type=str, help="Ubicacion de las carpetas con las imagenes",
                        default=def_img)

    args = parser.parse_args()
    MAIN_FOLDER = args.folder_csv
    MAIN_FOLDER_IMGS = args.folder_images
    print("Main Folder:", MAIN_FOLDER)
    print("Main Folder Imgs:", MAIN_FOLDER_IMGS)

    dict_dir_images = read_csv_data()
    k_fold(dict_dir_images)
