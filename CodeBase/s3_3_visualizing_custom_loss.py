from s2_1_LSTM import custom_loss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm


def visualize_loss_function_unit_test(loss_function, title, save_path):
    # Creating a grid of values
    range_values = np.linspace(0, 1, 100)
    high_pred, high_true = np.meshgrid(range_values, range_values)

    loss_values = np.zeros_like(high_pred)

    for i in tqdm(range(high_pred.shape[0])):
        for j in range(high_pred.shape[1]):
            y_true = np.array([[high_true[i, j], 0]])  
            y_pred = np.array([[high_pred[i, j], 0]])  
            loss_values[i, j] = loss_function(y_true, y_pred)

    print("Mean value of loss function: ", np.mean(loss_values))
    print("Max value of loss function: ", np.max(loss_values))
    print("Min value of loss function: ", np.min(loss_values))
    print("")

    plt.figure()
    plt.contourf(high_pred, high_true, loss_values, 40, cmap='plasma', vmin=0)
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('Target')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    visualize_loss_function_unit_test(custom_loss, 'Custom Loss Function', 'Visualizations/Loss_functions/custom_loss.png')

    mse_loss = lambda y_true, y_pred: np.mean(np.square(y_pred - y_true), axis=-1)
    visualize_loss_function_unit_test(mse_loss, 'Mean Squared Error', 'Visualizations/Loss_functions/mse_loss.png')