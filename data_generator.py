import numpy as np
import pandas as pd
import os.path

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from pandas.util.testing import assert_frame_equal
from scipy.ndimage.filters import gaussian_filter1d

def get_training_data(mergeMutated):
    print("Load Training Data...")
    train = pd.read_csv("data/train.csv")

    if mergeMutated == True and os.path.isfile("data/mutated_train.csv"):
        mutated_train = pd.read_csv("data/mutated_train.csv")

        train = pd.concat([train, mutated_train])

    return train


SCALE_FACTOR = 0.50 # was 0.25
MUTATE_RATE = 0.25 # 0.50 is too much , 0.25 has been optimal, try 0.3, 0.4
def mutate(x):

    should_mutate_vec = np.random.rand(x.size) < MUTATE_RATE
    scale_factor_x_vec = np.abs(x) * SCALE_FACTOR
    mutated_x_vec = x + scale_factor_x_vec * (np.random.rand(x.size) - 0.5)

    new_x = np.where(should_mutate_vec, mutated_x_vec, x)

    return new_x


FLIP_RATE = 0.00
GAUSS_FILTER_RATE = 0.00
def transform_df(orig_df, isInteractive=False):
    transformed_df = orig_df.copy(deep=True)

    # plot orig
    if isInteractive:
        pyplot.subplot(1, 4, 1)
        pyplot.imshow(transformed_df.iloc[:, 2:].as_matrix())

    transformed_df.iloc[:, 2:] = transformed_df.iloc[:, 2:].apply(mutate, axis=1)

    # plot mutations
    if isInteractive:
        pyplot.subplot(1, 4, 2)
        pyplot.imshow(transformed_df.iloc[:, 2:].as_matrix())


    if np.random.rand() < GAUSS_FILTER_RATE:
        transformed_df.iloc[:, 2:] = gaussian_filter1d(transformed_df.iloc[:, 2:].as_matrix(), sigma=1)

    # plot gaussian filter
    if isInteractive:
        pyplot.subplot(1, 4, 3)
        pyplot.imshow(transformed_df.iloc[:, 2:].as_matrix())

    def flip_feature_chunks(row):
        reshaped_row = row.reshape(3, 192/3)
        #print(reshaped_row)

        flipped_reshaped_row = np.fliplr(reshaped_row)
        #print(flipped_reshaped_row)

        return flipped_reshaped_row.reshape(192)

    if np.random.rand() < FLIP_RATE:
        m = transformed_df.iloc[:, 2:].as_matrix()
        transformed_df.iloc[:, 2:] = np.apply_along_axis(flip_feature_chunks, 1, m)

    # plot flip
    if isInteractive:
        pyplot.subplot(1, 4, 4)
        pyplot.imshow(transformed_df.iloc[:, 2:].as_matrix())

    # show the plot
    if isInteractive:
        pyplot.show()

    #assert_frame_equal(transformed_df, orig_df)

    return transformed_df


def build_mutated_dfs(orig_df, count, isInteractive=False):
    transformed_df_list = []
    for i in range(count):
        print("%d of %d" % (i, count))
        transformed_df_list.append(transform_df(orig_df, isInteractive))

    transformed_dfs = pd.concat(transformed_df_list)

    return transformed_dfs

def keras_data_generator(train_data, batch_size, growth_factor, isInteractive):
    data_generator = ImageDataGenerator(dim_ordering = 'tf')
    data_generator.fit(train_data)

    train_data_row_count = train_data.shape[0]
    total_requested = train_data_row_count * growth_factor

    X = train_data.iloc[:, 2:].as_matrix()
    X = np.expand_dims(X, axis=2)
    X = np.expand_dims(X, axis=1)

    # get original index
    Y = np.arange(X.shape[0])

    #Y = train_data.iloc[:, 1].as_matrix()
    #Y = np.expand_dims(Y, axis=1)

    X_gen_list = []
    Y_gen_list = []

    total_generated = 0
    for X_batch, Y_batch in data_generator.flow(X, Y, batch_size=batch_size):
        total_generated += batch_size

        current_batch_size = len(X_batch)
        for i in range(0, current_batch_size):

            X_batch_i = X_batch[i]
            X_batch_i = np.squeeze(X_batch_i, axis=(0,))
            X_batch_i = X_batch_i.reshape((3,np.int32(192/3)))

            X_i = X[[Y_batch[i]], 0, :].reshape((3,np.int32(192/3)))

            pyplot.subplot(current_batch_size, 2, (i*2) + 1)
            pyplot.imshow(X_batch_i) # cmap=pyplot.get_cmap('gray'))
            pyplot.subplot(current_batch_size, 2, (i*2) + 2)
            pyplot.imshow(X_i)# cmap=pyplot.get_cmap('gray'))

        # show the plot
        pyplot.show()

        X_gen_list.append(X_batch)
        Y_gen_list.append(Y_batch)

        if total_generated >= total_requested:
            break

    X_gen = np.vstack(X_gen_list)
    Y_gen = np.vstack(Y_gen_list)

def create_mutated_train_csv(train_data, growth_factor, isInteractive=False):

    mutated_dfs = build_mutated_dfs(train_data, growth_factor, isInteractive)

    mutated_dfs.to_csv('data/mutated_train.csv', index=False)

if __name__ == "__main__":
    train_data = get_training_data(mergeMutated=False)

    create_mutated_train_csv(train_data, growth_factor=100, isInteractive=False)
    #keras_data_generator(train_data, 32, 1)
