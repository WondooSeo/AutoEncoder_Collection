## Setup ##
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid the error "CPU supports"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # Avoid the error "Not creating XLA devices, tf_xla_enable_xla_devices not set"

## Define the VAE as a `Model` with a custom 'train_step' ##
class CAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }


if __name__ == '__main__':
    ## Make a dataset with shuffling ##
    path_dir = "./Your_Imgae_Directory"
    file_list = []
    for (root, directories, files) in os.walk(path_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    data_num = len(file_list)
    img_stacking = []
    count = 0

    for img in file_list:
        np_img = np.asarray(Image.open(img)) / 255.
        # tensor_img = tf.convert_to_tensor(np_img)
        # temp_stacking.append(tensor_img)
        img_stacking.append(np_img)
        count += 1
        print(str(count) + " / " + str(data_num) + " Stack Finished ...")

    img_stacking = np.random.permutation(img_stacking)
    img_stacking = np.expand_dims(img_stacking, -1)
    print("Pixel value normalize & shuffling Finished ...")

    ## Build the encoder ##
    encoder_path = "./encoder_Your_Name.h5"
    if (os.path.exists(encoder_path)):
        encoder = keras.models.load_model(encoder_path, compile=False)
        encoder.summary()
        print("Decoder model exist & loaded ...")

    else:
        # Input data : 128 X 128 X 1
        latent_dim = 20
        encoder_inputs = keras.Input(shape=(128, 128, 1))
        x = layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        z = layers.Dense(latent_dim, activation="relu")(x)
        encoder = Model(encoder_inputs, z, name="encoder")
        encoder.summary()

    ## Build the decoder ##
    decoder_path = "./decoder_Your_Name.h5"
    if (os.path.exists(decoder_path)):
        decoder = keras.models.load_model(decoder_path, compile=False)
        decoder.summary()
        print("Decoder model exist & loaded ...")

    else:
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((8, 8, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 8, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

    ## Train the CAE ##
    x_train, _ = train_test_split(img_stacking, test_size=0.3)
    cae = CAE(encoder, decoder)
    cae.compile(optimizer=keras.optimizers.Adam())
    cae.fit(x_train, epochs=300, batch_size=50)

    z_sample = [[1]*latent_dim]
    x_decoded = cae.decoder.predict(z_sample)
    plt.imshow(np.reshape(x_decoded, (128, 128, 1)))
    plt.axis('off')
    plt.show()

    ## Latent vector layer code ##
    # loc_z = len(encoder.layers) - 1
    # print(encoder.layers[loc_z].output_shape)

    # Save encoder
    encoder.save(encoder_path)
    print("Encoder model saved ... ")
    # Save decoder
    decoder.save(decoder_path)
    print("Decoder model saved ... ")
