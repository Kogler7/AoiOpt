import matplotlib.pyplot as plt


def plot_rewards(ys, names, file):
    fig = plt.figure(len(ys), figsize=[8 * len(ys), 10])

    for i in range(len(ys)):
        ax = fig.add_subplot(len(ys), 1, i + 1)
        ax.set_title(names[i])
        ax.plot(ys[i])

    plt.savefig(file)
    # plt.show()
    plt.close()