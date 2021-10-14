from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import pydicom
import glob
from skimage.restoration import denoise_nl_means, estimate_sigma


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


class DataGenerator:

    def __init__(self, batch_size, source_path, label_path, aug=False, seed=123, validation_split=0., subset='training'):
        '''
        Custom DataGenerator

        Parameters
        ----------
        batch_size : int
            batch size
        source_path : str
            path of the directory containing dicom files
        label_path : str
            path of the directory containing labels images
        aug : bool, optional
            data augmentation, by default False
        seed : int, optional
            random seed, by default 123
        validation_split : optional
            validation split rate, by default 0.
        subset : str, optional
            set training or validation subset of data, by default 'training'
        '''


        np.random.seed(seed)
        source_files = sorted(glob.glob(source_path + '/*.dcm'))
        source_files = np.asarray(source_files)


        labels_files = sorted(glob.glob(label_path + '/*.png'))
        labels_files = np.asarray(labels_files)

        assert source_files.size == labels_files.size


        source_files, labels_files = self.randomize(source_files, labels_files)

        idx = np.arange(0, source_files.size)
        np.random.shuffle(idx)

        self._source_trainfiles = source_files[idx[int(source_files.size * validation_split):]]
        self._labels_trainfiles = labels_files[idx[int(labels_files.size * validation_split):]]

        self._source_valfiles = source_files[idx[:int(source_files.size * validation_split)]]
        self._labels_valfiles = labels_files[idx[:int(labels_files.size * validation_split)]]

        self.subset = subset

        if self.subset == 'training':
            self._num_data = self._source_trainfiles.size
        elif self.subset == 'validation':
            self._num_data = self._source_valfiles.size

        self.aug = aug
        self._batch = batch_size
        self._cbatch = 0
        self._data, self._label = (None, None)

    @property
    def num_data(self):
        return self._num_data

    def randomize(self, source, label):

        random_index = np.arange(0, source.size)
        np.random.shuffle(random_index)
        source = source[random_index]
        label = label[random_index]

        return (source, label)


    def resize(self, img, lbl):

        height, width = img.shape[0], img.shape[1]

        if height != 512:
            img = cv2.resize(img, (512, 512))
            lbl = cv2.resize(lbl, (512, 512))
        else:
            img = img
            lbl = lbl

        return (img, lbl)


    def crop(self, img, lbl):

        height, width = img.shape[0], img.shape[1]

        if height != 512:
            img = cv2.resize(img, (512, 512))
            lbl = cv2.resize(lbl, (512, 512))

        assert img.shape[0] == 512

        y, x = 256, 256
        dy, dx = y // 2, x // 2

        return (img[(y - dy):(y + dy), (x - dx):(x + dx)], lbl[(y - dy):(y + dy), (x - dx):(x + dx)])

    def random_vflip(self, img, lbl):
        idx = np.random.uniform(low=0., high=1.)
        if idx > 0.5:
            return (cv2.flip(img, 0), cv2.flip(lbl, 0))
        else:
            return (img, lbl)

    def random_hflip(self, img, lbl):
        idx = np.random.uniform(low=0., high=1.)
        if idx > 0.5:
            return (cv2.flip(img, 1), cv2.flip(lbl, 1))
        else:
            return (img, lbl)

    def random_rotation(self, img, lbl):
        idx = np.random.uniform(low=0., high=1.)
        if idx > 0.5:
            def rotate(image, angle):
                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)

                # Perform the rotation
                M = cv2.getRotationMatrix2D(center, angle, 1.)
                rotated = cv2.warpAffine(image, M, (w, h))

                return rotated

            angle = np.random.randint(low=0, high=360)
            rotate_image = rotate(img, angle)
            rotate_lbl = rotate(lbl, angle)
            _, rotate_lbl = cv2.threshold(rotate_lbl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return (rotate_image, rotate_lbl)
        else:
            return (img, lbl)


    def rescale(self, img):
        rescaled = cv2.normalize(img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return rescaled

    def denoise(self, img):

        patch_kw = dict(patch_size=5, patch_distance=6)
        sigma_est = np.mean(estimate_sigma(img))
        denoised = denoise_nl_means(img, h=10 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)
        return denoised

    def gamma_correction(self, img, gamma=1.0):
        igamma = 1.0 / gamma
        imin, imax = img.min(), img.max()

        img_c = img.copy()
        img_c = ((img_c - imin) / (imax - imin)) ** igamma
        img_c = img_c * (imax - imin) + imin
        return img_c

    def __iter__(self):
        self._cbatch = 0
        return self

    def __next__(self):
        if self._cbatch + self._batch >= self._num_data:
            self._cbatch = 0
            self._source_trainfiles, self._labels_trainfiles = self.randomize(self._source_trainfiles, self._labels_trainfiles)
            self._source_valfiles, self._labels_valfiles = self.randomize(self._source_valfiles, self._labels_valfiles)

        if self.subset == 'training':
            c_sources = self._source_trainfiles[self._cbatch:self._cbatch + self._batch]
            c_labels = self._labels_trainfiles[self._cbatch:self._cbatch + self._batch]
        elif self.subset == 'validation':
            c_sources = self._source_valfiles[self._cbatch:self._cbatch + self._batch]
            c_labels = self._labels_valfiles[self._cbatch:self._cbatch + self._batch]

        # load the data

        images = [pydicom.dcmread(f).pixel_array for f in c_sources]
        labels = [cv2.imread(f, 0) for f in c_labels]

        # crop
        # images, labels = zip(*[self.crop(im, lbl) for im, lbl in zip(images, labels)])

        # check size

        images, labels = zip(*[self.resize(im, lbl) for im, lbl in zip(images, labels)])

        # cast

        images = [self.rescale(im) for im in images]
        labels = [self.rescale(lbl) for lbl in labels]

        # denoise
        images = [self.denoise(im) for im in images]

        # gamma correction
        images = [self.gamma_correction(im, gamma=1.5) for im in images]

        if self.aug:

            # random horizontal flip
            images, labels = zip(*[self.random_hflip(im, lbl) for im, lbl in zip(images, labels)])

            # random vertical flip
            images, labels = zip(*[self.random_vflip(im, lbl) for im, lbl in zip(images, labels)])

        images = [im[..., np.newaxis] for im in images]
        labels = [lbl[..., np.newaxis] for lbl in labels]

        # to numpy

        images = np.array(images)
        labels = np.array(labels)

        self._cbatch += self._batch

        return (images, labels)
