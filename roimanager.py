from read_roi import read_roi_file
import os
import cv2
import matplotlib.pyplot as plt
import argparse


def parse_args():

    description = 'Show ROIs of the original images'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--src', dest='src', required=True, type=str,
                        action='store', help='source directory')
    parser.add_argument('--patience', dest='patience', required=True, type=str,
                        action='store', help='Patience name', default='')
    parser.add_argument('--type', dest='type', required=True, type=str,
                        action='store', help='type of weight (i.e. T2 or DWI)', default='T2')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    if not os.path.isdir(args.src):
        raise ValueError('Incorrect directory given')

    roi_dir = os.path.join(args.src, args.patience, args.type + 'ROI')

    ROIs = []
    for item in os.listdir(roi_dir):

        path_to_roi = os.path.join(roi_dir, item)

        roi = read_roi_file(path_to_roi)

        ROIs.append(roi)

    def _dict(dict_list):
        '''

        useful to get true_dict since roi is {name file : true_dict}.

        '''

        true_dict = []

        for i in dict_list:
            _dict = list(i.values())

            for j in _dict:
                keys = j.keys()
                vals = j.values()

                _dict = {key: val for key, val in zip(keys, vals)}
                true_dict.append(_dict)

        return true_dict

    ROIs = _dict(ROIs)
    ROIs = sorted(ROIs, key=lambda d: list(d.values())[-1])  # ordering dictionaries by positions

    for roi in ROIs:
        position = roi['position']
        x = roi['x']
        y = roi['y']

        img_path = os.path.join(args.src, args.patience, args.type +
                                "_frames", str(position) + '.png')

        img = cv2.imread(img_path)

        plt.figure(1)
        plt.clf()
        plt.imshow(img)
        plt.plot(x, y, "red")
        plt.title("slice " + str(position))
        plt.pause(1)


if __name__ == '__main__':

    main()
