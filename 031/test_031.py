'''
Given an RGB image ("./test_image.png"), split it into three separate images, one for each channel (R, G, B) and calculate histograms with 256 bins for each channel. Save them as "red_channel_hist.npy", "green_channel_hist.npy", and "blue_channel_hist.npy".
'''

import cv2
from pathlib import Path
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt


def split_image(image_path:Path) -> List[np.ndarray]:
    image = cv2.imread(str(image_path))
    return cv2.split(image)

def calculate_histogram(channel:np.ndarray) -> np.ndarray:
    return cv2.calcHist([channel], [0], None, [256], [0, 256])

def save_histogram(hist:np.ndarray, filename:str):
    np.save(filename, hist)

def main():
    image_path = Path('./test_image.png')
    red_channel, green_channel, blue_channel = split_image(image_path)
    
    red_hist = calculate_histogram(red_channel)
    green_hist = calculate_histogram(green_channel)
    blue_hist = calculate_histogram(blue_channel)

    save_histogram(red_hist, 'red_channel_hist.npy')
    save_histogram(green_hist, 'green_channel_hist.npy')
    save_histogram(blue_hist, 'blue_channel_hist.npy')

def test():
    assert Path('red_channel_hist.npy').exists()
    assert Path('green_channel_hist.npy').exists()
    assert Path('blue_channel_hist.npy').exists()

    # load the histograms
    red_hist = np.load('red_channel_hist.npy')
    green_hist = np.load('green_channel_hist.npy')
    blue_hist = np.load('blue_channel_hist.npy')

    # load the image
    image = cv2.imread('test_image.png')
    red_channel, green_channel, blue_channel = cv2.split(image)

    # calculate the histograms
    red_hist_test = calculate_histogram(red_channel)
    green_hist_test = calculate_histogram(green_channel)
    blue_hist_test = calculate_histogram(blue_channel)

    # assert that the histograms are the same
    assert np.array_equal(red_hist, red_hist_test)
    assert np.array_equal(green_hist, green_hist_test)
    assert np.array_equal(blue_hist, blue_hist_test)

    # clean up
    # os.remove('red_channel_hist.npy')
    # os.remove('green_channel_hist.npy')
    # os.remove('blue_channel_hist.npy')

if __name__ == '__main__':
    # main()
    test()
    print('All tests passed')