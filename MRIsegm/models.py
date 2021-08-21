import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']

########################

# ! U-NET

########################


def unet(IMAGE_HEIGHT, IMAGE_WIDTH, n_levels=4, initial_features=64, n_conv=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1, activation="sigmoid"):
    '''
    U-Net is a convolutional neural network that was developed for biomedical image segmentation.The network consists of a contracting path and an expansive path, which gives it the u-shaped architecture. The contracting path is a typical convolutional network that consists of repeated application of convolutions, each followed by a rectified linear unit (ReLU) and a max pooling operation. During the contraction, the spatial information is reduced while feature information is increased. The expansive pathway combines the feature and spatial information through a sequence of up-convolutions and concatenations with high-resolution features from the contracting path.

    Parameters
    ----------
    IMAGE_HEIGHT : int
        height of the images.
    IMAGE_WIDTH : int
        width of the images.
    n_levels : int, optional
        number of contracting levels, by default 4.
    initial_features : int, optional
        number of initial convolutional layers, by default 64.
    n_conv : int, optional
        number of performed convolutions, by default 2.
    kernel_size : int, optional
        size of the kernel, by default 3.
    pooling_size : int, optional
        size of pooling, by default 2.
    in_channels : int, optional
        number of input channels, by default 1.
    out_channels : int, optional
        number of output channels, by default 1.
    activation: str
        keras activation function name.

    Returns
    -------
    keras Model class object
        U-net model.

    References
    -----------
    - Wiki: https://en.wikipedia.org/wiki/U-Net
    - U-Net architecture : '../extras/U-Net arch.jpeg'
    - U-Net paper: https://arxiv.org/pdf/1505.04597.pdf

    '''

    inputs = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    convpars_down = dict(kernel_size=kernel_size, activation='relu', padding='same')
    convpars_up = dict(kernel_size=kernel_size, activation='relu', padding='same')

    # downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_conv):
            x = Conv2D(initial_features * 2 ** level, **convpars_down)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = MaxPool2D(pooling_size)(x)

    # upstream
    for level in reversed(range(n_levels - 1)):
        x = Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars_up)(x)
        x = Concatenate()([x, skips[level]])
        for _ in range(n_conv):
            x = Conv2D(initial_features * 2 ** level, **convpars_up)(x)

    # output

    x = Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)

    return tf.keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')
