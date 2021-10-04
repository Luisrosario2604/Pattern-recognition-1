#!/usr/bin/python3

# Importing python3 from local, just use "python3 <binary>" if is not the same location

# Imports
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Global variables

print_shapes = True
show_image = True
np.random.seed(seed=123)  # <-to force that every run produces the same outcome (comment, or remove, to get randomness)


# Function declarations

def scale_to_unit(data):
    # Since all the pixels are in [0,255] we know the max and min for every possible pixel
    # --> so we can scale all the data at the same time
    # Usually we have to learn the max and min of the training set and save it to scale wrt them
    data = (data / 255.0)
    return data


def split_train_test(data, test_ratio):
    # data is the complete dataframe
    # test_ratio is in [0,1], and represents the percentage of the dataframe held for testing
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    test_set = data.iloc[test_indices]
    return train_set.reset_index(drop=True), test_set.reset_index(drop=True)


def show_three_and_seven(set1, set2):
    instance_id_to_show = 9  # <- index of the instance of 3 and 7 that will be shown in a figure

    # --- Plot the whole Data Sets
    f1, ax = plt.subplots(2, 2)  # 2, 2 size of the subplot
    ax[0, 0].imshow(set1, cmap='Blues')
    ax[0, 1].imshow(set2, cmap='Blues')
    # --> 2 figures in which each row is a row of the dataset

    # --- Plot an instance of set1
    ax[1, 0].imshow(set1.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    # --- Plot an instance of set2
    ax[1, 1].imshow(set2.iloc[instance_id_to_show].values.reshape(28, 28), 'Blues')
    plt.show()


def loading_datasets(location_three, location_seven):
    fraction_test = 0.2  # <- Percentage of the dataset held for test, in [0,1]
    fraction_valid = 0.1  # <- Percentage of the train set held for validation, in [0,1]

    full_set_3 = pd.read_csv(location_three, header=None)
    full_set_7 = pd.read_csv(location_seven, header=None)

    # --- Separate Test set -----------------------------
    train_set_3, test_set_3 = split_train_test(full_set_3, fraction_test)
    train_set_7, test_set_7 = split_train_test(full_set_7, fraction_test)

    # --- Separate Validation set -----------------------
    train_set_3, valid_set_3 = split_train_test(train_set_3, fraction_valid)
    train_set_7, valid_set_7 = split_train_test(train_set_7, fraction_valid)

    if print_shapes:
        print("Shape of full set 3 = ", full_set_3.shape)
        print("Shape of full set 7 = ", full_set_7.shape)

        print("\nShape of train set 3 = ", train_set_3.shape)
        print("Shape of train set 7 = ", train_set_7.shape)

        print("\nShape of test set 3 = ", test_set_3.shape)
        print("Shape of test set 7 = ", test_set_7.shape)

        print("\nShape of valid set 3 = ", valid_set_3.shape)
        print("Shape of valid set 7 = ", valid_set_7.shape)

        full_set_3 = scale_to_unit(full_set_3)
        full_set_7 = scale_to_unit(full_set_7)
        train_set_3 = scale_to_unit(train_set_3)
        train_set_7 = scale_to_unit(train_set_7)
        test_set_3 = scale_to_unit(test_set_3)
        test_set_7 = scale_to_unit(test_set_7)
        valid_set_3 = scale_to_unit(valid_set_3)
        valid_set_7 = scale_to_unit(valid_set_7)

    return full_set_3, full_set_7, train_set_3, train_set_7, test_set_3, test_set_7, valid_set_3, valid_set_7


def main():
    full_set_3,  \
    full_set_7,  \
    train_set_3, \
    train_set_7, \
    test_set_3,  \
    test_set_7,  \
    valid_set_3, \
    valid_set_7 = loading_datasets('./Datasets/1000_tres.csv', './Datasets/1000_siete.csv')

    if show_image:
        show_three_and_seven(train_set_3, train_set_7)


# Main body
if __name__ == '__main__':
    main()
