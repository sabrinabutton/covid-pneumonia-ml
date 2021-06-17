import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import kerastuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def generate_data():
    training_data_generator = ImageDataGenerator(
        rescale=1.0/255)

    training_iterator = training_data_generator.flow_from_directory(
        "data/train", class_mode="categorical", color_mode="grayscale", target_size=(256, 256), batch_size=1)

    testing_iterator = training_data_generator.flow_from_directory(
        "data/test", class_mode="categorical", color_mode="grayscale", target_size=(256, 256), batch_size=1)

    return training_iterator, testing_iterator


def generate_model(tuner):
    model = Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    
    convunits = tuner.Int('conv2d_units', min_value=6, max_value=32, step=2)
    strides = tuner.Int('strides', min_value=1, max_value=5, step=1)
    size = tuner.Int('size', min_value=1, max_value=5, step=1)
    model.add(tf.keras.layers.Conv2D(convunits, size,  strides= strides, activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(size, size), strides= strides,padding="same"))
    model.add(tf.keras.layers.Conv2D(convunits, size, strides= strides, activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(size, size), strides= strides,padding="same"))

    model.add(layers.Flatten())

    denseunits = tuner.Int('dense_units', min_value=6, max_value=16, step=2)
    model.add(tf.keras.layers.Dense(denseunits, activation='sigmoid'))

    model.add(layers.Dense(3, activation='softmax'))

    learning_rate = tuner.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(
    ), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

    return model


def generate_plot(history):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(history.history['categorical_accuracy'])
    ax1.plot(history.history['val_categorical_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(history.history['auc'])
    ax2.plot(history.history['val_auc'])
    ax2.set_title('model auc')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('auc')
    ax2.legend(['train', 'validation'], loc='upper left')
    fig.tight_layout()
    plt.savefig('static/my_plots.png')


tuner = kt.Hyperband(generate_model,
                     objective='val_categorical_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hyperparam-tuning',
                     project_name='covid-pneumonia-ml')
training_iterator, testing_iterator = generate_data()
stop = EarlyStopping(monitor='val_loss', patience=5)

tuner.search(training_iterator, validation_data=testing_iterator, epochs=50,
           callbacks=[stop])

#Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
#model = generate_model()
history = model.fit(training_iterator,validation_data=testing_iterator, epochs=10)

val_acc_per_epoch = history.history['val_categorical_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
history = model.fit(training_iterator,validation_data=testing_iterator,  epochs=best_epoch)

eval_result = model.evaluate(testing_iterator)
print("[test loss, test accuracy]:", eval_result)
generate_plot(history)

# Generate classification report and confusion matrix
test_steps_per_epoch = np.math.ceil(testing_iterator.samples / testing_iterator.batch_size)
predictions = model.predict(testing_iterator, steps=test_steps_per_epoch)
test_steps_per_epoch = np.math.ceil(testing_iterator.samples / testing_iterator.batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = testing_iterator.classes
class_labels = list(testing_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)
print(cm)