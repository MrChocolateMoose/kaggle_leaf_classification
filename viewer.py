from matplotlib import pyplot as plt

def show_margin(X, index):
    margin_mat = X[index,:].reshape(1,64)

    plt.imshow(margin_mat, interpolation='nearest')
    plt.show()


def show_shape(X, index):
    shape_mat = X[index,:].reshape(1,64)

    plt.imshow(shape_mat, interpolation='nearest')
    plt.show()

def show_all(X, index):
    all_mat = X[index,:].reshape(3,64)

    plt.imshow(all_mat, interpolation='nearest')
    plt.show()

def show_all_rows(X):
    for index in range(X.shape[0]):
        show_margin(index)
        show_shape(index)
        show_all(index)

