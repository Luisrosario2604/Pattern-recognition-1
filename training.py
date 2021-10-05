#!/usr/bin/python3

# Importing python3 from local, just use "python3 <binary>" if is not the same location

# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

# Global variables

print_shapes = False
show_image = False
show_extractions = True
np.random.seed(seed=123)  # <-to force that every run produces the same outcome (comment, or remove, to get randomness)
z_general = []
x1_max = 0
x2_max = 0
three_general = []
seven_general = []
save_model = False


# Function declarations


def jitter(x, sigma=0.06):
    random_sign = (-1)**np.random.randint(1, 3, *x.shape)
    return x + np.random.normal(0, sigma, * x.shape) * random_sign


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


def feat_extraction(data, label, generation_global = False,theta=0.6):

    global z_general
    global x1_max
    global x2_max
    global three_general
    global seven_general

    # data: dataframe
    # theta: parameter of the feature extraction
    #
    features = np.zeros([data.shape[0], 9])  # <- allocate memory with zeros
    data = data.values.reshape([data.shape[0], 28, 28])
    # -> axis 0: id of instance, axis 1: width(cols) , axis 2: height(rows)
    for k in range(data.shape[0]):
        # ..current image
        x = data[k, :, :]
        # --width feature
        sum_cols = x.sum(axis=0)  # <- axis0 of x, not of data!!
        indc = np.argwhere(sum_cols > theta * sum_cols.max())
        col_3maxs = np.argsort(sum_cols)[-3:]
        features[k, 0] = indc[-1] - indc[0]
        features[k, 1:4] = col_3maxs
        # --width feature
        sum_rows = x.sum(axis=1)  # <- axis1 of x, not of data!!
        indr = np.argwhere(sum_rows > theta * sum_rows.max())
        features[k, 4] = indr[-1] - indr[0]
        row_3maxs = np.argsort(sum_rows)[-3:]
        features[k, 5:8] = row_3maxs

        all_pixels = sum(sum_rows)
        sum_rows_tmp = 0
        num_rows_reach_half = 0
        for num_of_pixels_row in sum_rows:
            if sum_rows_tmp < all_pixels / 3:
                sum_rows_tmp += num_of_pixels_row
                num_rows_reach_half += 1

        sum_rows_tmp1 = 0
        for tmp in range(7):
            if sum_rows[tmp] > theta * sum_rows[tmp].max():
                sum_rows_tmp1 += 1

        features[k, 8] = sum_rows_tmp1

        x1 = (features[k, 4] ** (1/3)) / 2.7
        x2 = (features[k, 8]) / 7.0

        if x1 > x1_max and generation_global:
            x1_max = x1
        if x2 > x2_max and generation_global:
            x2_max = x2
        if generation_global:
            z_general.append([x1, x2, label])
        if label == 0 and generation_global:
            three_general.append([x1, x2])
        elif generation_global:
            seven_general.append([x1, x2])

    col_names = ['width', 'W_max1', 'W_max2', 'W_max3', 'height', 'H_max1', 'H_max2', 'H_max3', 'number_pixels_seven_rows']
    return pd.DataFrame(features, columns=col_names)


def test_model(model, test_set_3, test_set_7):
    extraction_test_set_3 = feat_extraction(test_set_3, 0)
    extraction_test_set_7 = feat_extraction(test_set_7, 1)

    col_names = ['x1', 'x2', 'label']
    test_set = pd.DataFrame(extraction_test_set_3, columns=col_names)

    #print(test_set)


def show_extraction_function(extraction3, extraction7):
    f1, ax = plt.subplots(2, 4)  # 2, 2 size of the subplot
    ax[0, 0].plot(jitter(extraction3['width']), "o", color='blue', ms=0.7)  # ms = marker size
    ax[0, 0].plot(jitter(extraction7['width']), "x", color='red', ms=0.7)
    ax[0, 0].title.set_text("width")

    ax[0, 1].plot(jitter(extraction3['height']), "o", color='blue', ms=0.7)
    ax[0, 1].plot(jitter(extraction7['height']), "x", color='red', ms=0.7)
    ax[0, 1].title.set_text("height")

    ax[0, 2].plot(jitter(extraction3['H_max1']), "o", color='blue', ms=0.7)
    ax[0, 2].plot(jitter(extraction7['H_max1']), "x", color='red', ms=0.7)
    ax[0, 2].title.set_text("W_max1")

    ax[1, 0].plot(jitter(extraction3['H_max2']), "o", color='blue', ms=0.7)
    ax[1, 0].plot(jitter(extraction7['H_max2']), "x", color='red', ms=0.7)
    ax[1, 0].title.set_text("W_max2")

    ax[1, 1].plot(jitter(jitter(extraction3['height']) ** (1/3)), "o", color='blue', ms=0.7)
    ax[1, 1].plot(jitter(jitter(extraction7['height']) ** (1/3)), "x", color='red', ms=0.7)
    ax[1, 1].title.set_text("Caract 1")
    
    calc1 = (extraction3['number_pixels_seven_rows'])
    calc2 = (extraction7['number_pixels_seven_rows'])

    ax[1, 2].plot(jitter(calc1), "o", color='blue', ms=0.7)
    ax[1, 2].plot(jitter(calc2), "x", color='red', ms=0.7)
    ax[1, 2].title.set_text("Caract 2")

    ax[0, 3].plot(jitter(three_general['x1']), "o", color='blue', ms=0.7)
    ax[0, 3].plot(jitter(seven_general['x1']), "x", color='red', ms=0.7)
    ax[0, 3].title.set_text("Carac 1.1")

    ax[1, 3].plot(jitter(three_general['x2']), "o", color='blue', ms=0.7)
    ax[1, 3].plot(jitter(seven_general['x2']), "x", color='red', ms=0.7)
    ax[1, 3].title.set_text("Carac 2.2")

    # idea de la cajas (dividido por 3)
    plt.show()


def show_image_function(set1, set2):
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

    full_set_3 = pd.read_csv(location_three, header=None)
    full_set_7 = pd.read_csv(location_seven, header=None)

    # --- Separate Test set -----------------------------
    train_set_3, test_set_3 = split_train_test(full_set_3, fraction_test)
    train_set_7, test_set_7 = split_train_test(full_set_7, fraction_test)

    if print_shapes:
        print("Shape of full set 3 = ", full_set_3.shape)
        print("Shape of full set 7 = ", full_set_7.shape)

        print("\nShape of train set 3 = ", train_set_3.shape)
        print("Shape of train set 7 = ", train_set_7.shape)

        print("\nShape of test set 3 = ", test_set_3.shape)
        print("Shape of test set 7 = ", test_set_7.shape)

        full_set_3 = scale_to_unit(full_set_3)
        full_set_7 = scale_to_unit(full_set_7)
        train_set_3 = scale_to_unit(train_set_3)
        train_set_7 = scale_to_unit(train_set_7)
        test_set_3 = scale_to_unit(test_set_3)
        test_set_7 = scale_to_unit(test_set_7)

    return full_set_3, full_set_7, train_set_3, train_set_7, test_set_3, test_set_7


def main():

    global z_general
    global three_general
    global seven_general

    full_set_3,  \
    full_set_7,  \
    train_set_3, \
    train_set_7, \
    test_set_3,  \
    test_set_7 = loading_datasets('./Datasets/1000_tres.csv', './Datasets/1000_siete.csv')

    if show_image:
        show_image_function(train_set_3, train_set_7)

    extraction_train_set_3 = feat_extraction(train_set_3, 0, True)
    extraction_train_set_7 = feat_extraction(train_set_7, 1, True)

    col_names = ['x1', 'x2', 'label']
    col_names_reduced = ['x1', 'x2']
    z_general = pd.DataFrame(z_general, columns=col_names)
    three_general = pd.DataFrame(three_general, columns=col_names_reduced)
    seven_general = pd.DataFrame(seven_general, columns=col_names_reduced)

    if show_extractions:
        show_extraction_function(extraction_train_set_3, extraction_train_set_7)

    # print(str(z_general) + "\n")
    # print("X1 MAX =" + str(x1_max))
    # print("X2 MAX =" + str(x2_max))

    model = LogisticRegression()
    model.fit(z_general.iloc[:, [0, 1]], z_general.iloc[:, 2])

    if save_model:
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    test_model(model, test_set_3, test_set_7)


# Main body
if __name__ == '__main__':
    main()
