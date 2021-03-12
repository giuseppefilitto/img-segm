from read_roi import read_roi_file
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
import argparse


def parse_args():

    description = 'Show ROIs of the original images'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--src', dest='src', required=True, type=str,
                        action='store', help='source directory')
    parser.add_argument('--patience', dest='patience', required=True, type=str,
                        action='store', help='Patience name', default='')
    parser.add_argument('--weight', dest='weight', required=True, type=str,
                        action='store', help='weight (i.e. T2 or DWI)', default='T2')

    args = parser.parse_args()

    return args


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


def main():

    args = parse_args()

    if not os.path.isdir(args.src):
        raise ValueError('Incorrect directory given')

    roi_dir = os.path.join(args.src, args.patience, args.weight + 'ROI')

    toggle = 0
    if os.path.isdir(roi_dir + "alta" or roi_dir + "bassa"):
        inp = input("alta o bassa?: ")
        roi_dir = roi_dir + str(inp)
        toggle = 1

    ROIs = []
    for item in os.listdir(roi_dir):

        path_to_roi = os.path.join(roi_dir, item)

        roi = read_roi_file(path_to_roi)

        ROIs.append(roi)

    ROIs = _dict(ROIs)  # get true_dict list
    # ordering dictionaries by positions
    ROIs = sorted(ROIs, key=lambda d: list(d.values())[-1])

    # filtering rois with no coordinates
    ROIs = list(filter(lambda d: d['type'] != 'composite', ROIs))

    for i in range(len(ROIs)):

        position = ROIs[i]['position']
        x = ROIs[i]['x']
        y = ROIs[i]['y']

        # to connect first and last point of the ROI
        x.append(x[0])
        y.append(y[0])

        img_dir = os.path.join(args.src, args.patience, args.weight)

        if not os.path.isdir(img_dir):
            img_dir = img_dir + "AX"

            if toggle == 1:
                img_dir = img_dir + inp

            if not os.path.isdir(img_dir):
                img_dir = os.path.join(args.src, args.patience, args.weight)
                img_dir = img_dir + "5mm"

        img_path = os.path.join(img_dir, str(position - 1) + '.dcm')
        img = pydicom.dcmread(img_path).pixel_array
        img = np.asarray(img)

        plt.figure(1)
        plt.clf()
        plt.imshow(img[:, :], cmap="gray")
        plt.axis('off')
        plt.plot(x, y, color='red', linestyle='dashed', linewidth=1)
        plt.title("slice " + str(position))
        plt.pause(1)


if __name__ == '__main__':

    main()
