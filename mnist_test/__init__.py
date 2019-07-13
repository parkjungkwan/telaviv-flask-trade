from mnist_test.mnist_load import MnistTest

if __name__ == '__main__':
    mtest = MnistTest()
    model = mtest.create_model()
    # i, predictions_array, true_label, img
    predictions_array =  model[0]
    true_label = model[1]
    img = model[2]
    mtest.plot_image(0, predictions_array, true_label, img)
