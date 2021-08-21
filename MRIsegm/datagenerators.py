from tensorflow.keras.preprocessing.image import ImageDataGenerator


__author__ = ['Giuseppe Filitto']
__email__ = ['giuseppe.filitto@studio.unibo.it']


def create_segmentation_generator(img_path, mask_path, BATCH_SIZE, IMG_SIZE, SEED, data_gen_args_img, data_gen_args_mask):
    '''

    Create DataGenerator yielding tuples of (x, y) with shape (batch_size, height, width, channels) where x is the input image and y is the ground-truth. The data generation is performed using data_gen_args_img and data_gen_args_mask.

    Parameters
    ----------
    img_path : str
        path for the images directory.
    mask_path : str
        path for the ground-truth directory.
    BATCH_SIZE : int
        size of the batches of data.
    IMG_SIZE : tuple
        (image height, image width).
    SEED : int
        seed for randomness control.
    data_gen_args_img: dict
        dict of keras ImageDataGenerator args for the generation of custom images.
    data_gen_args_mask: dict
        dict of keras ImageDataGenerator args for the generation of custom masks.
    Returns
    -------
    zip object
        tuples of (x, y) with shape (batch_size, height, width, channels) where x is the input image and y is the ground-truth.

    '''

    img_data_gen = ImageDataGenerator(**data_gen_args_img)
    mask_data_gen = ImageDataGenerator(**data_gen_args_mask)

    img_generator = img_data_gen.flow_from_directory(
        img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    mask_generator = mask_data_gen.flow_from_directory(
        mask_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)

    return zip(img_generator, mask_generator)
