import matplotlib.pyplot as plt
import numpy as np


def plot_train_loss_folds(train_losses_all):
    plt.figure(figsize=(9, 6))

    for i, train_losses in enumerate(train_losses_all):
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f'Fold {i+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epoch (All Folds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_val_loss_folds(val_losses_all):
    plt.figure(figsize=(9, 6))

    for i, val_losses in enumerate(val_losses_all):
        epochs = np.arange(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, label=f'Fold {i+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch (All Folds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_val_dice_folds(val_dices_all):
    plt.figure(figsize=(9, 6))

    for i, val_dices in enumerate(val_dices_all):
        epochs = np.arange(1, len(val_dices) + 1)
        plt.plot(epochs, val_dices, label=f'Fold {i+1}')

    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice vs Epoch (All Folds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

