import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error

import math

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

try:
    from Scripts.Read_FileImage import ubuntu, window_size, MODEL_DEF
    from Scripts.utils import BATCH_SIZE
except:
    from Read_FileImage import ubuntu, window_size, MODEL_DEF
    from utils import BATCH_SIZE

import matplotlib.pyplot as plt

if ubuntu:
    def_directory = "/home/folder_def/Documents/main_dir/Scripts/modelos"
else:
    def_directory = "/main_dir/Scripts/modelos"


def get_number_from_image(array_images):
    labels_number_images = [int(x[0].split("_")[1].split(".")[0]) for x in array_images[0]]
    if not isinstance(labels_number_images, list):
        if not isinstance(labels_number_images, np.ndarray):
            labels_number_images = np.concatenate(labels_number_images, axis=0)
        images_list = labels_number_images.tolist()
    else:
        images_list = labels_number_images
    return images_list


def get_predictions(test_data, labels_data, labels_main_images, labels_all_images, FPN_model, index=[0],
                    model_dir_file=None, MaxQtyTraining=42, loop=None):
    try:
        from Scripts.utils import configurations_FPN
    except:
        from utils import configurations_FPN

    global IMG_DEF_HEIGHT, IMG_DEF_WIDTH

    if model_dir_file:
        FPN_model.load_weights(model_dir_file)

    predictions = np.zeros(test_data.shape[0])
    if ubuntu:
        predictions = FPN_model.predict(test_data, batch_size=BATCH_SIZE)

    predictions *= MaxQtyTraining
    predictions = np.around(predictions, 2)

    predictions_list = predictions.tolist()
    labels_list = (np.around(labels_data * MaxQtyTraining, 2)).tolist()

    all_images_labels = get_number_from_image(labels_all_images)
    main_images_labels = get_number_from_image(labels_main_images)

    index_list = np.full(labels_data.shape, index[0]).tolist()
    loop_list = np.full(labels_data.shape, loop).tolist()

    result_array = np.column_stack((predictions_list, labels_list, main_images_labels, all_images_labels, index_list,
                                    loop_list))

    return result_array


def save_file_predictions(predictions_obj, model_name="", name_sheet="Test", data_type=""):
    if not isinstance(predictions_obj, pd.DataFrame):
        columns = ['Predictions', 'Ground_Truth', 'Main Image', 'Side Images', 'Partition', 'Loop']
        df = pd.DataFrame(predictions_obj, columns=columns)
        df = df.astype({"Main Image": "int", "Side Images": "int", "Partition": "int", "Loop": "int"})
        df['Loop'] = df['Loop'].replace({None: np.nan})
        df['Key'] = df['Loop'].astype(str) + '_' + df['Partition'].astype(str)
        df['Error'] = round(abs(df['Predictions'] - df['Ground_Truth']), 2)

        df_error_stats = df.groupby('Key')['Error'].agg(
            ['mean', 'var']).reset_index()  # agrupacion por key y calculo de mean y var de la columna error
        df_error_stats.columns = ['Key', 'Mean Error', 'Var Error']
        df = pd.merge(df, df_error_stats, on='Key')

        df['Mean Error'] = round(abs(df['Mean Error']), 2)
        df['Var Error'] = round(abs(df['Var Error']), 2)
    else:
        df = predictions_obj

    try:
        df['Predictions'] = df['Predictions'].round(2)
        df['Ground_Truth'] = df['Ground_Truth'].round(2)
    except:
        pass

    if ubuntu:
        dir_docs = '/home/folder_def/Documents/main_dir/Scripts/Docs/'
        if data_type:
            dir_docs += '{}/'.format(data_type)
        if model_name:
            dir_docs += '{}/'.format(model_name)
        if window_size:
            dir_docs += 'window_{}/'.format(window_size)

        if not os.path.exists(dir_docs):
            os.makedirs(dir_docs)

        try:
            with pd.ExcelWriter(dir_docs + '{}_metrics.xlsx'.format(model_name), mode='a',
                                if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=name_sheet, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(dir_docs + '{}_metrics.xlsx'.format(model_name), mode='x') as writer:
                df.to_excel(writer, sheet_name=name_sheet, index=False)
    else:
        print(df.head())

    return df


def get_metrics_from_predictions(df, by_tree=False, model_name="", data_type=""):
    columns = ['Partition', 'Image', 'Mean_Predictions', 'Mean_Labels', 'MAE', 'Error', 'Desviacion Estandar',
               'Max_Value',
               'Min_Value',
               ]
    df_result = pd.DataFrame(columns=columns)
    unique_partition_values = df['Key'].unique()
    if 'Key' not in df.columns:
        new_row_df = pd.DataFrame({'Partition': "-",
                                   'Image': "-",
                                   'Mean_Predictions': 0,
                                   'Mean_Labels': 0,
                                   'MAE': round(mean_absolute_error(df['Ground_Truth'],
                                                                    df['Predictions']), 2),
                                   'Error': 0,
                                   'Desviacion Estandar': 0,
                                   'Max_Value': 0,
                                   'Min_Value': 0,
                                   },
                                  index=[0])
        df_result = pd.concat([df_result, new_row_df], ignore_index=True)
    else:
        for partition in unique_partition_values:
            df_partition = df[df['Key'] == partition]
            if data_type == "TEST" and not by_tree:
                if model_name:
                    folder = "{}/{}".format(model_name, str(partition))
                else:
                    folder = "{}".format(str(partition))
                calculate_distributions(df_partition, folder=folder, title="{}_{}".format(window_size, str(partition)))

            if by_tree:
                df_result_by_partition = pd.DataFrame(columns=columns)
                unique_images_values = df_partition['Main Image'].unique()
                for tree in unique_images_values:
                    df_by_tree = df_partition[df_partition['Main Image'] == tree]
                    num_rows, num_cols = df_by_tree.shape
                    if num_rows > 0:
                        mean_predictions = df_by_tree['Predictions'].mean()
                        mean_real = df_by_tree['Ground_Truth'].mean()
                        std_deviation = round(np.std(df_by_tree['Predictions'] - df_by_tree['Ground_Truth']), 2)
                        max_value = round(df_by_tree['Predictions'].max(), 2)
                        min_value = round(df_by_tree['Predictions'].min(), 2)
                        new_row_df = pd.DataFrame({'Partition': partition,
                                                   'Image': tree,
                                                   'Mean_Predictions': round(mean_predictions, 2),
                                                   'Mean_Labels': round(mean_real, 2),
                                                   'MAE': round(mean_absolute_error(df_by_tree['Ground_Truth'],
                                                                                    df_by_tree['Predictions']), 2),
                                                   'Error': round(abs(mean_real - mean_predictions), 2),
                                                   'Desviacion Estandar': std_deviation,
                                                   'Max_Value': max_value,
                                                   'Min_Value': min_value
                                                   }, index=[0])
                        df_result = pd.concat([df_result, new_row_df], ignore_index=True)
                        df_result_by_partition = pd.concat([df_result_by_partition, new_row_df], ignore_index=True)

                df_result_by_partition["Median"] = round(df_by_tree['Predictions'].median(), 2)
                df_result_by_partition["Mean Error"] = round(df_result_by_partition["Error"].mean(), 2)
                df_result_by_partition["Var Error"] = round(df_result_by_partition["Error"].var(), 2)

                save_file_predictions(df_result_by_partition,
                                      model_name=model_name,
                                      name_sheet="Key_{}_by_Tree".format(partition),
                                      data_type=data_type)
            else:
                new_row_df = pd.DataFrame({'Partition': partition,
                                           'Image': "ALL",
                                           'Mean_Predictions': 0,
                                           'Mean_Labels': 0,
                                           'MAE': round(mean_absolute_error(df_partition['Ground_Truth'],
                                                                            df_partition['Predictions']), 2),
                                           'Error': 0,
                                           'Desviacion Estandar': 0,
                                           'Max_Value': 0,
                                           'Min_Value': 0,
                                           },
                                          index=[0])
                df_result = pd.concat([df_result, new_row_df], ignore_index=True)

        if not by_tree:
            save_file_predictions(df_result, model_name=model_name, name_sheet="Metric_per_partitions",
                                  data_type=data_type)

    return df_result


def create_graphs(df_results, by_tree=False, title_img=None):
    # Crea una nueva columna que combina 'Index' e 'Image' para el eje x del gráfico
    df_results['Partition_Image'] = df_results['Partition'].astype(str) + "_" + df_results['Image'].astype(str)

    bar_width = 0.35
    plt.clf()
    plt.figure(figsize=(10, 6))

    if by_tree:
        unique_partition_values = df_results['Partition'].unique()
        for partition in unique_partition_values:
            plt.clf()
            df_partition = df_results[df_results['Partition'] == partition]
            max_pred = int(max(df_partition['Mean_Predictions']))
            max_real = int(max(df_partition['Mean_Labels']))
            max_value = max(max_pred, max_real)
            plt.ylim(0, max_value + 5)
            x = np.arange(len(df_partition['Partition_Image']))

            # Crea un gráfico de barras para 'Median_Predictions' y 'Median_Labels'
            plt.bar(x - bar_width / 2, df_partition['Mean_Predictions'], bar_width, alpha=0.5,
                    label='Mean_Predictions')
            plt.ylim(0, 35)
            plt.yticks(range(0, 35, 5))
            plt.bar(x + bar_width / 2, df_partition['Mean_Labels'], bar_width, alpha=0.5, label='Mean_Labels')
            plt.ylabel('Mean Value')
            plt.xlabel('Image')
            plt.xticks(x, df_partition['Image'], rotation=90)
            if title_img:
                plt.title(
                    "{}, Loop:{}, Partition:{}".format(title_img, partition.split("_")[0], partition.split("_")[-1]))
            plt.legend()
            plt.tight_layout()
            if not title_img:
                title_img = "test"
            try:
                if ubuntu:
                    dir_image = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/{}/'.format(
                        title_img.replace(" ", "_"))
                else:
                    dir_image = '/main_dir/Scripts/Imagenes/{}/'.format(title_img.replace(" ", "_"))

                if MODEL_DEF:
                    dir_image += "{}/".format(str(MODEL_DEF).strip())

                if window_size:
                    dir_image += "window_{}/".format(window_size)

                if not os.path.exists(dir_image):
                    os.makedirs(dir_image)

                plt.savefig(dir_image + '{}_{}.png'.format(partition.split("_")[0], partition.split("_")[-1]))
            except Exception as e:
                print("ERROR CREANDO IMAGEN POR: {}".format(str(e)))
                pass
    else:
        plt.ylim(0, 15)
        x = np.arange(len(df_results['Partition_Image']))
        plt.bar(x, df_results['MAE'], alpha=0.5, label='MAE')
        plt.ylabel('MAE')
        plt.xlabel('Loop_Partition')
        plt.xticks(x, df_results['Partition'], rotation=90)
        if title_img:
            plt.title(title_img)
        plt.legend()
        plt.tight_layout()

        try:
            if ubuntu:
                dir_image = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/{}/'.format(
                    title_img.replace(" ", "_"))
            else:
                dir_image = '/main_dir/Scripts/Imagenes/{}/'.format(title_img.replace(" ", "_"))

            if MODEL_DEF:
                dir_image += "{}/".format(str(MODEL_DEF).strip())

            if window_size:
                dir_image += "window_{}/".format(window_size)

            if not os.path.exists(dir_image):
                os.makedirs(dir_image)

            plt.savefig(dir_image + 'MAE.png')
        except Exception as e:
            print("ERROR CREANDO IMAGEN POR: {}".format(str(e)))
            pass


def create_files_with_results(results_training, results_test):
    predictions_training = np.vstack(results_training)
    predictions_test = np.vstack(results_test)

    df_train = save_file_predictions(predictions_training, model_name=MODEL_DEF, name_sheet="All_Predictions_Train",
                                     data_type="TRAIN")
    df_test = save_file_predictions(predictions_test, model_name=MODEL_DEF, name_sheet="All_Predictions_Test",
                                    data_type="TEST")

    print("*" * 10)
    print("Metricas por Arbol - Datos Entrenamiento")
    df_results_by_tree = get_metrics_from_predictions(df_train, by_tree=True, model_name=MODEL_DEF, data_type="TRAIN")
    create_graphs(df_results_by_tree, by_tree=True, title_img="Train Dataset")

    print("*" * 10)
    print("Metricas por Fotograma - Datos Entrenamiento")
    df_results_grl = get_metrics_from_predictions(df_train, model_name=MODEL_DEF, data_type="TRAIN")
    create_graphs(df_results_grl, title_img="Train Dataset")

    print("*" * 10)
    print("Metricas por Arbol - Datos Prueba")
    df_results_by_tree = get_metrics_from_predictions(df_test, by_tree=True, model_name=MODEL_DEF, data_type="TEST")
    create_graphs(df_results_by_tree, by_tree=True, title_img="Test Dataset")

    print("*" * 10)
    print("Metricas por Fotograma - Datos Prueba")
    df_results_grl = get_metrics_from_predictions(df_test, model_name=MODEL_DEF, data_type="TEST")
    create_graphs(df_results_grl, title_img="Test Dataset")


def box_plot_save(title, data, data_labels):
    plt.clf()
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(data, labels=data_labels)
    plt.xticks(rotation=45)
    plt.xlabel('Frame')
    plt.ylabel('Values')
    plt.title('Boxplot Partition: {}'.format(title))

    try:
        if ubuntu:
            dir_boxplot = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/BoxPlot/'
        else:
            dir_boxplot = '/main_dir/Scripts/Imagenes/BoxPlot/'

        if MODEL_DEF:
            dir_boxplot += "{}/".format(str(MODEL_DEF).strip())

        if window_size:
            dir_boxplot += "window_{}/".format(window_size)

        if not os.path.exists(dir_boxplot):
            os.makedirs(dir_boxplot)

        plt.savefig(dir_boxplot + "{}.png".format(title.replace(" ", "_")))
    except Exception as e:
        print("ERROR CREANDO BOXPLOT POR: {}".format(str(e)))
        pass


def calculate_uncertainty_by_range(group):
    max_value = group['Predictions'].max()
    min_value = group['Predictions'].min()
    return round((max_value - min_value) / 2, 2)


def calculate_uncertainty_by_stdev(group):
    deviation = round(np.std(group['Predictions'], ddof=1), 2)
    return deviation


# El dataframe debe tener las columans Main_Image, Ground_Truth, Predictions
def calculate_distributions(df, folder="0", title="0"):
    data = []
    data_labels = []

    # MAE at frame level
    df['difference'] = abs(df['Ground_Truth'] - df['Predictions'])
    tree_group = df.groupby('Main Image')  # Agrupacion por Main_Image
    grouped = tree_group['difference'].sum().reset_index()  # Suma de las diferencias por cada grupo de arboles
    qty_trees = len(grouped)  # cantidad de arboles
    qty_frames = tree_group['Main Image'].count().iloc[0]  # cantidad de fotogramas por arboles
    total_mean = grouped['difference'].sum() / (
            qty_frames * qty_trees)  # Suma de todas las diferencias y division por el total de instancias

    # MAE at sequence level
    grouped_means_difference = abs(tree_group['Ground_Truth'].mean() - tree_group['Predictions'].mean())
    total_mean_sequence = grouped_means_difference.sum() / (qty_trees)

    grouped_median_difference = abs(tree_group['Ground_Truth'].median() - tree_group['Predictions'].median())
    total_median_sequence = grouped_median_difference.sum() / (qty_trees)

    mae_data = pd.DataFrame({
        'MAE_frame': [total_mean],
        'MAE_sequence/mean': [total_mean_sequence],
        'MAE_sequence/median': [total_median_sequence]
    })

    # Uncertainty
    result_df = tree_group.apply(calculate_uncertainty_by_range).reset_index(name='By_Range')
    result_df_deviation = tree_group.apply(calculate_uncertainty_by_stdev).reset_index(name='By_Stdv')

    final_result = result_df.merge(result_df_deviation, on='Main Image')
    final_result["Uncertainty_By_Range"] = round(final_result['By_Range'] / math.sqrt(qty_frames), 2)
    final_result["Uncertainty_By_Stdv"] = round(final_result['By_Stdv'] / math.sqrt(qty_frames), 2)

    serie1 = final_result['By_Range'].values.tolist()
    serie2 = final_result['Uncertainty_By_Range'].values.tolist()
    serie3 = final_result['By_Stdv'].values.tolist()
    serie4 = final_result['Uncertainty_By_Stdv'].values.tolist()

    box_plot_save("Series " + title, [serie1, serie2, serie3, serie4], ["Range", "SEM - Range", "Stdv", "Sem - Stdv"])

    for name_f in final_result["Main Image"]:
        predictions = df[df["Main Image"] == name_f]["Predictions"].values.tolist()

        if not data:
            data = [predictions]
        else:
            data.append(predictions)
        data_labels.append("rgb_{}".format(str(name_f)))

    box_plot_save(title, data, data_labels)

    if ubuntu:
        dir_docs = '/home/folder_def/Documents/main_dir/Scripts/Docs/Distributions/'
        dir_docs += '{}/'.format(folder)
        if not os.path.exists(dir_docs):
            os.makedirs(dir_docs)

        try:
            with pd.ExcelWriter(dir_docs + 'Distributions.xlsx', mode='a',
                                if_sheet_exists="replace") as writer:
                final_result.to_excel(writer, sheet_name=title, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(dir_docs + 'Distributions.xlsx', mode='x') as writer:
                final_result.to_excel(writer, sheet_name=title, index=False)

        try:
            with pd.ExcelWriter(dir_docs + 'MAE_corrected.xlsx', mode='a',
                                if_sheet_exists="replace") as writer:
                mae_data.to_excel(writer, sheet_name=title, index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(dir_docs + 'MAE_corrected.xlsx', mode='x') as writer:
                mae_data.to_excel(writer, sheet_name=title, index=False)
    else:
        print(final_result.head())


def scatter_plot_results(df_results, title_img="Test", model_name="", max_val_data=None):
    if not isinstance(df_results, pd.DataFrame):
        df_results = pd.DataFrame(df_results,
                                  columns=['Predictions', 'Ground_Truth', 'Main Image', 'Side Images', 'Partition',
                                           'Loop'])
        df_results = df_results.astype({"Main Image": "int", "Side Images": "int", "Partition": "int", "Loop": "int"})
        df_results['Loop'] = df_results['Loop'].replace({None: np.nan})
        df_results['Key'] = df_results['Loop'].astype(str) + '_' + df_results['Partition'].astype(str)
        try:
            df_results['Predictions'] = df_results['Predictions'].round(2)
            df_results['Ground_Truth'] = df_results['Ground_Truth'].round(2)
        except:
            pass

    df_results['Error_scatter'] = abs(df_results['Ground_Truth'] - df_results['Predictions'])
    fig2, ax2 = plt.subplots()
    ax2.scatter(df_results['Ground_Truth'], df_results["Error_scatter"])
    if max_val_data:
        ax2.set_ylim(0, int(max_val_data))
    else:
        ax2.set_ylim(0, 70)
    try:
        if ubuntu:
            dir_scatter = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/Scatter_Error/'
        else:
            dir_scatter = '/main_dir/Scripts/Imagenes/Scatter_Error/'

        if model_name:
            dir_scatter += '{}/'.format(model_name)

        if window_size:
            dir_scatter += 'window_{}/'.format(window_size)

        if not os.path.exists(dir_scatter):
            os.makedirs(dir_scatter)

        fig2.savefig(dir_scatter + '{}.png'.format(title_img.replace(" ", "_")))
    except Exception as e:
        print("ERROR CREANDO IMAGEN POR: {}".format(str(e)))
        pass


def histogram_samples(y_train, y_test, min_limit, max_limit, train_index=[0], test_index=[0]):
    # TRAIN
    plt.clf()
    intervalos = range(int(min_limit), int(max_limit) + 5, 5)
    plt.subplot(1, 2, 1)

    hist_values, hist_bins, _ = plt.hist(x=y_train, bins=intervalos, color='#F2AB6D', rwidth=1)
    max_hist_value = max(hist_values)
    plt.title('Train Group - Partitions: {}'.format(train_index))
    plt.xlabel('Cantidad Naranjas')
    plt.ylabel('Frecuencia')
    plt.xticks(intervalos)
    plt.ylim(0, int(max_hist_value) + 5)

    plt.subplot(1, 2, 2)
    plt.hist(x=y_test, bins=intervalos, rwidth=1)
    plt.title('Test Group - Partition: {}'.format(test_index[0]))
    plt.xlabel('Cantidad Naranjas')
    plt.ylabel('Frecuencia')
    plt.xticks(intervalos)
    plt.ylim(0, int(max_hist_value) + 5)
    plt.tight_layout()
    try:
        if ubuntu:
            dir_hist = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/Histograms/'
        else:
            dir_hist = '/main_dir/Scripts/Imagenes/Histograms/'

        if MODEL_DEF:
            dir_hist += "{}/".format(str(MODEL_DEF).strip())

        if window_size:
            dir_hist += "window_{}/".format(window_size)

        if not os.path.exists(dir_hist):
            os.makedirs(dir_hist)

        print("SAVING HIST AT: ", dir_hist + 'Figure_{}.png'.format(test_index[-1]))
        plt.savefig(dir_hist + 'Figure_{}.png'.format(test_index[-1]))
    except Exception as e:
        print("ERROR CREANDO HISTOGRAMA POR: {}".format(str(e)))
        pass


def graph_training_result(model_history, title="0"):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(model_history.history['loss'], label='training')
    plt.plot(model_history.history['val_loss'], label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper right')

    try:
        if ubuntu:
            dir_boxplot = '/home/folder_def/Documents/main_dir/Scripts/Imagenes/Training_Curves/'
        else:
            dir_boxplot = '/main_dir/Scripts/Imagenes/Training_Curves/'

        if MODEL_DEF:
            dir_boxplot += "{}/".format(str(MODEL_DEF).strip())

        if window_size:
            dir_boxplot += "window_{}/".format(window_size)

        if not os.path.exists(dir_boxplot):
            os.makedirs(dir_boxplot)

        plt.savefig(dir_boxplot + "{}.png".format(title))
    except Exception as e:
        print("ERROR CREANDO BOXPLOT POR: {}".format(str(e)))
        pass


def save_model_history(model_history=None, dir_train_index=None):
    if model_history:
        if ubuntu:
            dir_hist = '/home/folder_def/Documents/main_dir/Scripts/Hist_Models/'
        else:
            dir_hist = '/main_dir/Scripts/Hist_Models/'

        if MODEL_DEF:
            dir_hist += "{}/".format(str(MODEL_DEF).strip())

        if window_size:
            dir_hist += "window_{}/".format(window_size)

        if dir_train_index:
            dir_hist += "{}/".format(dir_train_index)

        if not os.path.exists(dir_hist):
            os.makedirs(dir_hist)

        # Guardar el historial en un archivo CSV
        with open(dir_hist + 'historial_entrenamiento.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Escribe el encabezado con los nombres de las métricas, incluyendo "epoch"
            fieldnames = ['epoch'] + list(model_history.history.keys())
            csv_writer.writerow(fieldnames)

            # Escribe los valores correspondientes a cada métrica en cada época, incluyendo el número de época
            for epoch, epoch_data in enumerate(zip(*model_history.history.values()), start=1):
                csv_writer.writerow([epoch] + list(epoch_data))
