## Setup ##
import os

import tensorflow.keras.backend
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.activations import relu
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
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


def sampling(mu_log_var):
    mu, log_var = mu_log_var
    sampling_epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tensorflow.keras.backend.exp(log_var/2) * sampling_epsilon
    return random_sample


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000 # This value is used only for vizual comfort
        reconstruction_loss = tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_true-y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tensorflow.keras.backend.sum(1.0 + encoder_log_variance - tensorflow.keras.backend.square(encoder_mu) - tensorflow.keras.backend.exp(encoder_log_variance), axis=[1, 2, 3])
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)
        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


if __name__ == '__main__':
    ## This is necessary for running VAE!!! ##
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    ## This is necessary for running VAE!!! ##

    latent_dim = 10
    path_dir = 'your path'

    ## Stacking a dataset
    file_list = stacking_files_dir(path_dir=path_dir)
    img_stacking = stacking_images(file_list=file_list)

    img_stacking = np.expand_dims(img_stacking, -1)
    x_train, x_test = train_test_split(img_stacking, shuffle=True, test_size=0.15)
    print("Pixel value normalize & shuffling Finished ...")

    ## VAE Encoder
    encoder_input = Input((128, 128, 1), name='encoder_input')

    encoder_conv_layer1 = Conv2D(filters=8, kernel_size=4, strides=2, padding='same', name='encoder_conv_layer1')(encoder_input)
    encoder_batch_layer1 = BatchNormalization(name='encoder_batch_layer1')(encoder_conv_layer1)
    encoder_active_layer1 = relu(encoder_batch_layer1)

    encoder_conv_layer2 = Conv2D(filters=16, kernel_size=4, strides=2, padding='same', name='encoder_conv_layer2')(encoder_active_layer1)
    encoder_batch_layer2 = BatchNormalization(name='encoder_batch_layer2')(encoder_conv_layer2)
    encoder_active_layer2 = relu(encoder_batch_layer2)

    encoder_conv_layer3 = Conv2D(filters=16, kernel_size=4, strides=2, padding='same', name='encoder_conv_layer3')(encoder_active_layer2)
    encoder_batch_layer3 = BatchNormalization(name='encoder_batch_layer3')(encoder_conv_layer3)
    encoder_active_layer3 = relu(encoder_batch_layer3)

    encoder_conv_layer4 = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', name='encoder_conv_layer4')(encoder_active_layer3)
    encoder_batch_layer4 = BatchNormalization(name='encoder_batch_layer4')(encoder_conv_layer4)
    encoder_active_layer4 = relu(encoder_batch_layer4)

    shape_last_conv_layer = tensorflow.keras.backend.int_shape(encoder_active_layer4)[1:]
    encoder_flatten = Flatten()(encoder_batch_layer4)
    encoder_mu = Dense(units=latent_dim, name='encoder_mu')(encoder_flatten)
    encoder_log_var = Dense(units=latent_dim, name='encoder_log_var')(encoder_flatten)
    encoder_log_var_model = Model(encoder_input, (encoder_mu, encoder_log_var), name='encoder_log_var_model')
    encoder_output = tensorflow.keras.layers.Lambda(sampling, name='encoder_output')([encoder_mu, encoder_log_var])
    encoder = Model(encoder_input, encoder_output, name='encoder_model')
    # encoder.summary()

    ## VAE Decoder
    decoder_input = Input(latent_dim, name='decoder_input')
    decoder_dense_layer1 = Dense(units=np.prod(shape_last_conv_layer), name='decoder_dense_layer1')(decoder_input)
    decoder_reshape = Reshape(target_shape=shape_last_conv_layer)(decoder_dense_layer1)

    decoder_convt_layer1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', name='decoder_convt_layer1')(decoder_reshape)
    decoder_batch_layer1 = BatchNormalization(name='decoder_batch_layer1')(decoder_convt_layer1)
    decoder_active_layer1 = relu(decoder_batch_layer1)

    decoder_convt_layer2 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='same', name='decoder_convt_layer2')(decoder_active_layer1)
    decoder_batch_layer2 = BatchNormalization(name='decoder_batch_layer2')(decoder_convt_layer2)
    decoder_active_layer2 = relu(decoder_batch_layer2)

    decoder_convt_layer3 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding='same', name='decoder_convt_layer3')(decoder_active_layer2)
    decoder_batch_layer3 = BatchNormalization(name='decoder_batch_layer3')(decoder_convt_layer3)
    decoder_active_layer3 = relu(decoder_batch_layer3)

    decoder_convt_layer4 = Conv2DTranspose(filters=8, kernel_size=4, strides=2, padding='same', name='decoder_convt_layer4')(decoder_active_layer3)
    decoder_batch_layer4 = BatchNormalization(name='decoder_batch_layer4')(decoder_convt_layer4)
    decoder_active_layer4 = relu(decoder_batch_layer4)

    decoder_convt_layer5 = Conv2DTranspose(filters=1, kernel_size=4, padding='same', name='decoder_convt_layer5')(decoder_active_layer4)
    decoder_output = sigmoid(decoder_convt_layer5)

    decoder = Model(decoder_input, decoder_output, name='decoder_model')
    # decoder.summary()

    ## Making meta-data of CAE layers
    VAE_input = Input(shape=(128, 128, 1), name='VAE_input')
    VAE_encoder_output = encoder(VAE_input)
    VAE_decoder_output = decoder(VAE_encoder_output)
    VAE = Model(VAE_input, VAE_decoder_output, name='VAE')
    VAE.summary()
    print('VAE model construction complete ...')

    VAE.compile(optimizer='adam', loss=loss_func(encoder_mu, encoder_log_var), metrics=['mae', 'mse'])
    history = VAE.fit(x_train, x_train, validation_split=0.15, epochs=30, batch_size=50, verbose=1, shuffle=True)

    # Latent vector code
    z_sample = [[1] * latent_dim]
    x_decoded = decoder.predict(z_sample)
    plt.imshow(np.reshape(x_decoded, (128, 128, 1)))
    plt.axis('off')
    plt.show()

    # Show plot of loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('VAE Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    test_scores = VAE.evaluate(x_test, verbose=0)
    # print(VAE.metrics_names) # → ['loss', 'mae', 'mse']
    print("Test Loss : ", test_scores[0])

    encoder_path = 'your encoder path.h5'
    decoder_path = 'yout decoder path.h5'
    encoder.save(encoder_path)
    decoder.save(decoder_path)
    print('Encoder & Decoder saved...')



