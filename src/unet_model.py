from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

def conv_block(x, num_filters: int):
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(x)
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal")(x)
    return x

def encoder_block(x, num_filters: int):
    s = conv_block(x, num_filters)
    p = MaxPooling2D((2, 2))(s)
    return s, p

def decoder_block(x, skip, num_filters: int):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = concatenate([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return Model(inputs, outputs, name="U-Net")
