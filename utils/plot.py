import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losses):
    plt.plot(np.arange(1, len(train_losses) + 1), [[train_losses[i], valid_losses[i]] for i in range(len(train_losses))])
    plt.legend(["Training loss", "Validation loss"])