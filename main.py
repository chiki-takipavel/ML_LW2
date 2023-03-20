import os
import cv2
import random
import logging
import datetime
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

DATASET_FOLDER = r'C:/Users/chiki/Downloads/notMNIST_large'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
CLASSES_COUNT = len(CLASSES)
DATA_COLUMN_NAME = 'data'
LABELS_COLUMN_NAME = 'labels'
HASHED_DATA_COLUMN_NAME = 'data_bytes'
BALANCE_BORDER = 0.85
TRAIN_SIZE = 200000
VALIDATION_SIZE = 10000
TEST_SIZE = 19000
BATCH_SIZE = 48
EPOCHS = 30
EPOCHS_RANGE = range(EPOCHS)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def show_one_image(image_folder):
    file = random.choice(os.listdir(image_folder))
    image_path = os.path.join(image_folder, file)
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.show()


def show_some_images():
    for class_item in CLASSES:
        image_folder = os.path.join(DATASET_FOLDER, class_item)
        show_one_image(image_folder)


def get_class_data(folder_path):
    result_data = list()
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)
        if img is not None:
            result_data.append(img)

    return result_data


def get_classes_images_counts(data_frame):
    classes_images_counts = list()
    for class_index in range(CLASSES_COUNT):
        labels = data_frame[LABELS_COLUMN_NAME]
        class_rows = data_frame[labels == class_index]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)
        logging.info(f"Class {CLASSES[class_index]} contains {class_count} images")

    return classes_images_counts


def create_data_frame():
    data = list()
    labels = list()
    for class_item in CLASSES:
        class_folder_path = os.path.join(DATASET_FOLDER, class_item)
        class_data = get_class_data(class_folder_path)

        data.extend(class_data)
        labels.extend([CLASSES.index(class_item) for _ in range(len(class_data))])

    data_frame = pd.DataFrame({DATA_COLUMN_NAME: data, LABELS_COLUMN_NAME: labels})
    logging.info("Data frame is created")

    return data_frame


def remove_duplicates(data):
    data_bytes = [item.tobytes() for item in data[DATA_COLUMN_NAME]]
    data[HASHED_DATA_COLUMN_NAME] = data_bytes
    data.sort_values(HASHED_DATA_COLUMN_NAME, inplace=True)
    data.drop_duplicates(subset=HASHED_DATA_COLUMN_NAME, keep='first', inplace=True)
    data.pop(HASHED_DATA_COLUMN_NAME)
    logging.info("Duplicates removed")

    return data


def show_classes_histogram(classes_images_counts):
    plt.figure()
    plt.bar(CLASSES, classes_images_counts)
    plt.show()
    logging.info("Histogram shown")


def check_classes_balance(data_frame):
    classes_images_counts = get_classes_images_counts(data_frame)

    max_images_count = max(classes_images_counts)
    avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
    balance_percent = avg_images_count / max_images_count

    show_classes_histogram(classes_images_counts)
    logging.info(f"Balance: {balance_percent:.3f}")
    if balance_percent > BALANCE_BORDER:
        logging.info("Classes are balanced")
    else:
        logging.info("Classes are not balanced")


def shuffle_data(data):
    data_shuffled = data.sample(frac=1, random_state=42)
    logging.info("Data shuffled")

    return data_shuffled


def split_dataset_into_subsamples(data_frame):
    data = list(data_frame[DATA_COLUMN_NAME].values)
    labels = list(data_frame[LABELS_COLUMN_NAME].values)

    data_dataset = tf.data.Dataset.from_tensor_slices(data)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((data_dataset, labels_dataset))

    train_dataset = dataset.take(TRAIN_SIZE).batch(BATCH_SIZE)
    validation_dataset = dataset.skip(TRAIN_SIZE).take(VALIDATION_SIZE).batch(BATCH_SIZE)
    test_dataset = dataset.skip(TRAIN_SIZE + VALIDATION_SIZE).take(TEST_SIZE).batch(BATCH_SIZE)
    logging.info("Data split")

    return train_dataset, validation_dataset, test_dataset


def get_statistics(model, train_dataset, validation_dataset, test_dataset, with_optimization=False):
    if with_optimization:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        optimizer = tf.keras.optimizers.experimental.SGD()

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    model_history = model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        verbose=1
    )

    loss, accuracy = model.evaluate(test_dataset)
    logging.info(f"Model: {accuracy=}, {loss=}")

    accuracy = model_history.history['accuracy']
    validation_accuracy = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    validation_loss = model_history.history['val_loss']

    return loss, accuracy, validation_loss, validation_accuracy


def get_neural_network_statistics(train_dataset, validation_dataset, test_dataset):
    losses = list()
    accuracies = list()
    validation_losses = list()
    validation_accuracies = list()

    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    simple_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
    ])

    regularized_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
    ])

    dynamic_model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASSES_COUNT, activation='softmax')
    ])

    simple_model_stats = get_statistics(
        simple_model, train_dataset, validation_dataset, test_dataset
    )

    regularized_model_stats = get_statistics(
        regularized_model, train_dataset, validation_dataset, test_dataset
    )

    dynamic_model_stats = get_statistics(
        dynamic_model, train_dataset, validation_dataset, test_dataset, with_optimization=True
    )

    losses.extend((simple_model_stats[0], regularized_model_stats[0], dynamic_model_stats[0]))
    accuracies.extend((simple_model_stats[1], regularized_model_stats[1], dynamic_model_stats[1]))
    validation_losses.extend((simple_model_stats[2], regularized_model_stats[2], dynamic_model_stats[2]))
    validation_accuracies.extend((simple_model_stats[3], regularized_model_stats[3], dynamic_model_stats[3]))

    return losses, accuracies, validation_losses, validation_accuracies


def show_result_plot(losses, accuracies, validation_losses, validation_accuracies):
    plt.figure(figsize=(14, 10))

    plt.subplot(1, 2, 1)
    plt.title('Training and Validation Loss')
    plt.plot(EPOCHS_RANGE, losses[0], label='Train Smpl Loss')
    plt.plot(EPOCHS_RANGE, validation_losses[0], label='Val Smpl Loss', linestyle='dashed')
    plt.plot(EPOCHS_RANGE, losses[1], label='Train Reg Loss')
    plt.plot(EPOCHS_RANGE, validation_losses[1], label='Val Reg Loss', linestyle='dashed')
    plt.plot(EPOCHS_RANGE, losses[2], label='Train Dyn Loss')
    plt.plot(EPOCHS_RANGE, validation_losses[2], label='Val Dyn Loss', linestyle='dashed')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.title('Training and Validation Accuracy')
    plt.plot(EPOCHS_RANGE, accuracies[0], label='Train Smpl Acc')
    plt.plot(EPOCHS_RANGE, validation_accuracies[0], label='Val Smpl Acc', linestyle='dashed')
    plt.plot(EPOCHS_RANGE, accuracies[1], label='Train Reg Acc')
    plt.plot(EPOCHS_RANGE, validation_accuracies[1], label='Val Reg Acc', linestyle='dashed')
    plt.plot(EPOCHS_RANGE, accuracies[2], label='Train Dyn Acc')
    plt.plot(EPOCHS_RANGE, validation_accuracies[2], label='Val Dyn Acc', linestyle='dashed')
    plt.legend(loc='upper right')

    plt.show()
    logging.info("Plot shown")


def main():
    start_time = datetime.datetime.now()

    show_some_images()

    data_frame = create_data_frame()
    data_frame = remove_duplicates(data_frame)
    check_classes_balance(data_frame)
    data_frame = shuffle_data(data_frame)

    train_dataset, validation_dataset, test_dataset = split_dataset_into_subsamples(data_frame)

    losses, accuracies, validation_losses, validation_accuracies = get_neural_network_statistics(
        train_dataset, validation_dataset, test_dataset
    )
    show_result_plot(losses, accuracies, validation_losses, validation_accuracies)

    end_time = datetime.datetime.now()
    logging.info(end_time - start_time)


main()
