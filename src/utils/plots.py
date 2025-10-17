import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_graph(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Perda no Treino")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Perda na Validação")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Vistos")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()