## Setup ##
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split


def stacking_files_dir(path_dir):
    file_list = []
    for (root, directories, files) in os.walk(path_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def stacking_images(file_list):
    data_num = len(file_list)
    img_stacking = list()
    count = 0

    for img in file_list:
        np_img = np.asarray(Image.open(img)) / 255.
        img_stacking.append(np_img)
        count += 1
        print(str(count) + " / " + str(data_num) + " Stack Finished ...")

        # Debugging code
        # if count == 1000:
        #     break

    return img_stacking


def CAE_encoder(latent_dim):
    encoder = Sequential(name='encoder')
    encoder.add(Conv2D(filters=4, kernel_size=3, strides=2, activation='relu', padding='same', input_shape=(128, 128, 1)))
    encoder.add(Conv2D(filters=8, kernel_size=3, strides=2, activation='relu', padding='same'))
    encoder.add(BatchNormalization())
    encoder.add(Conv2D(filters=16, kernel_size=3, strides=2, activation='relu', padding='same'))
    encoder.add(Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'))
    encoder.add(BatchNormalization())
    encoder.add(Flatten())
    encoder.add(Dense(latent_dim))
    # encoder.summary()
    return encoder


def CAE_decoder():
    decoder = Sequential(name='decoder')
    decoder.add(Dense(8 * 8 * 32, activation='relu', input_shape=(latent_dim,)))
    decoder.add(Reshape((8, 8, 32)))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='relu', padding='same'))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='relu', padding='same'))
    decoder.add(Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation='relu', padding='same'))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same'))
    # decoder.summary()
    return decoder


if __name__ == '__main__':
    ## This is necessary for running CAE!!! ##
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    ## This is necessary for running CAE!!! ##

    latent_dim = 10
    image_method = 'GREIT'
    path_dir = './EIT_' + image_method

    ## Stacking a dataset
    file_list = stacking_files_dir(path_dir=path_dir)
    img_stacking = stacking_images(file_list=file_list)

    img_stacking = np.expand_dims(img_stacking, -1)
    x_train, x_test = train_test_split(img_stacking, shuffle=True, test_size=0.15)
    print("Pixel value normalize & shuffling Finished ...")

    ## Create encoder and decoder model
    encoder = CAE_encoder(latent_dim=latent_dim)
    decoder = CAE_decoder()

    ## Making meta-data of CAE layers
    CAE_input = Input(shape=(128, 128, 1), name='CAE_input')
    CAE_encoder_output = encoder(CAE_input)
    CAE_decoder_output = decoder(CAE_encoder_output)
    CAE = Model(CAE_input, CAE_decoder_output, name='CAE')
    CAE.summary()

    CAE.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae', 'mse', 'accuracy'])
    history = CAE.fit(x_train, x_train, validation_split=0.15, epochs=100, batch_size=300, verbose=1, shuffle=True)

    # Latent vector code
    z_sample = [[0.5] * latent_dim]
    x_decoded = decoder.predict(z_sample)
    plt.imshow(np.reshape(x_decoded, (128, 128, 1)))
    plt.axis('off')
    plt.show()

    # Show plot of loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CAE Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    test_scores = CAE.evaluate(x_test, verbose=0, batch_size=10)
    # print(CAE.metrics_names) # → ['loss', 'mae', 'mse', 'accuracy']
    print("Test Loss : ", test_scores[0])
    print("Test Accuracy : ", test_scores[-1])

    encoder_path = './encoder_'+str(latent_dim)+'_'+image_method+'.h5'
    decoder_path = './decoder_'+str(latent_dim)+'_'+image_method+'.h5'
    encoder.save(encoder_path)
    decoder.save(decoder_path)
