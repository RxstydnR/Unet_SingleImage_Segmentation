import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Input, Dropout, MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D, Conv2DTranspose, Concatenate, LeakyReLU
from tensorflow.keras.models import Model

class UNet(tf.keras.Model):
    """
        U-Net model
    """
    def __init__(self, input_ch, output_ch, filters, INPUT_IMAGE_SIZE=256):
        super().__init__()

        self.INPUT_IMAGE_SIZE = INPUT_IMAGE_SIZE
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # (256 x 256 x input_channel_count)
        inputs = Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_ch))

        """
            Encoder
        """
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(filters, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        enc2 = self._add_encoding_layer(filters*2, enc1)

        # (32 x 32 x 4N)
        enc3 = self._add_encoding_layer(filters*4, enc2)

        # (16 x 16 x 8N)
        enc4 = self._add_encoding_layer(filters*8, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(filters*8, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(filters*8, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filters*8, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filters*8, enc7)

        """
            Decoder
        """
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filters*8, True, enc8)
        dec1 = Concatenate(axis=-1)([dec1, enc7])

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filters*8, True, dec1)
        dec2 = Concatenate(axis=-1)([dec2, enc6])

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filters*8, True, dec2)
        dec3 = Concatenate(axis=-1)([dec3, enc5])

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filters*8, False, dec3)
        dec4 = Concatenate(axis=-1)([dec4, enc4])

        # (32 x 32 x 4N)
        dec5 = self._add_decoding_layer(filters*4, False, dec4)
        dec5 = Concatenate(axis=-1)([dec5, enc3])

        # (64 x 64 x 2N)
        dec6 = self._add_decoding_layer(filters*2, False, dec5)
        dec6 = Concatenate(axis=-1)([dec6, enc2])

        # (128 x 128 x N)
        dec7 = self._add_decoding_layer(filters, False, dec6)
        dec7 = Concatenate(axis=-1)([dec7, enc1])

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(output_ch, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(inputs=inputs, outputs=dec8)

    def _add_encoding_layer(self, filters, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filters, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filters, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filters, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET