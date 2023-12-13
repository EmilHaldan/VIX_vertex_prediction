import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def combine_all_metrics(ml_models_dir):
    all_mectrics_dict = {"model_name" : []}
    for model_name in os.listdir(ml_models_dir):
        if len(os.listdir(f"{ml_models_dir}/{model_name}")) != 5:
            continue
        cur_metrics_df = pd.read_csv(f"{ml_models_dir}/{model_name}/{model_name}_metrics.csv")
        for col in cur_metrics_df.columns:
            if col in all_mectrics_dict.keys():
                all_mectrics_dict[col].append(cur_metrics_df.loc[0,col])
            else: 
                all_mectrics_dict[col] = [cur_metrics_df.loc[0,col]]
        all_mectrics_dict["model_name"].append(model_name)

    all_metrics_df = pd.DataFrame(all_mectrics_dict)

    return all_metrics_df


def print_metrics_for_best_model(all_metrics_df, cur_dir):
    best_model_idx = all_metrics_df["val_loss"].idxmin()
    best_model_name = all_metrics_df.loc[best_model_idx, "model_name"]
    print("\n"*2)
    print("best_model_name                   : ", best_model_name)
    print("")
    print("val_loss                          : ", round(all_metrics_df.loc[best_model_idx, "val_loss"], 4))
    print("val_mse                           : ", round(all_metrics_df.loc[best_model_idx, "val_mse"], 4))
    print("val_mae                           : ", round(all_metrics_df.loc[best_model_idx, "val_mae"], 4))
    print("")
    print("test_loss                         : ", round(all_metrics_df.loc[best_model_idx, "test_loss"], 4))
    print("test_mse                          : ", round(all_metrics_df.loc[best_model_idx, "test_mse"], 4))
    print("test_mae                          : ", round(all_metrics_df.loc[best_model_idx, "test_mae"], 4))
    print("epochs_trained                    : ", all_metrics_df.loc[best_model_idx, "epochs_trained"])
    print("\n"*2)


def aggregate_metrics(all_metrics):
    """
    Aggregates metrics and returns a DataFrame with statistics.
    """
    metrics = ['val_loss','test_loss','test_mse','val_mse','val_mae','test_mae','epochs_trained']

    stats = {
        'mean': all_metrics.mean(),
        'median': all_metrics.median(),
        'std': all_metrics.std(),
        'min': all_metrics.min(),
        'max': all_metrics.max()
    }

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.loc[metrics]
    new_stats_df = stats_df.transpose()

    return new_stats_df


def plot_training_loss(ml_models_dir, loss_type='loss'):
    """
    Plots the training and validation loss (MSE or Loss) for models in the specified directory.

    :param ml_models_dir: Directory containing ML model subdirectories.
    :param loss_type: Type of loss to plot ('loss' or 'MSE').
    """
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    best_model_training_loss = None
    best_model_validation_loss = None

    for model_name in os.listdir(ml_models_dir):
        if len(os.listdir(f"{ml_models_dir}/{model_name}")) != 5:
            continue
        model_path = os.path.join(ml_models_dir, model_name)
        if os.path.isdir(model_path):
            log_file = os.path.join(model_path, 'training_log.csv')

            try:
                log_data = pd.read_csv(log_file, delimiter=';', usecols=['epoch', loss_type, f'val_{loss_type}'])
                training_loss = log_data[loss_type].to_list()
                validation_loss = log_data[f'val_{loss_type}'].to_list()

                if validation_loss[-1] < best_val_loss:
                    best_val_loss = validation_loss[-1]
                    best_model_training_loss = training_loss
                    best_model_validation_loss = validation_loss

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
            except FileNotFoundError:
                print(f"Log file not found in {model_path}")

    if not training_losses or not validation_losses:
        print("No loss data found.")
        return

    max_len = max(len(l) for l in training_losses + validation_losses)

    training_losses = [l + [np.nan] * (max_len - len(l)) for l in training_losses]
    validation_losses = [l + [np.nan] * (max_len - len(l)) for l in validation_losses]

    if best_model_training_loss is not None and best_model_validation_loss is not None:
        best_model_training_loss += [np.nan] * (max_len - len(best_model_training_loss))
        best_model_validation_loss += [np.nan] * (max_len - len(best_model_validation_loss))

    avg_training_loss = np.array([safe_nanmean(column) for column in np.array(training_losses).T])
    std_training_loss = np.array([safe_nanstd(column) for column in np.array(training_losses).T])
    avg_validation_loss = np.array([safe_nanmean(column) for column in np.array(validation_losses).T])
    std_validation_loss = np.array([safe_nanstd(column) for column in np.array(validation_losses).T])

    # Plotting
    epochs = range(1, len(avg_training_loss) + 1)
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, avg_training_loss, label='Average Training Loss', color = "#04a3c7")
    plt.fill_between(epochs, avg_training_loss - std_training_loss, avg_training_loss + std_training_loss, alpha=0.2, color = '#04a3c7')

    plt.plot(epochs, avg_validation_loss, label='Average Validation Loss', color='#f58105')
    plt.fill_between(epochs, avg_validation_loss - std_validation_loss, avg_validation_loss + std_validation_loss, alpha=0.2, color='#f58105')

    if best_model_training_loss is not None and best_model_validation_loss is not None:
        plt.plot(epochs, best_model_training_loss, label='Best Model Training Loss', color='#0004ff')
        plt.plot(epochs, best_model_validation_loss, label='Best Model Validation Loss', color='#f52505')

    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_type.capitalize()} Value')
    plt.title(f"{ml_models_dir.replace('ML_saved_models/', '')}  Training and Validation {loss_type.capitalize()} Over Epochs")

    plt.xlim(9, int(max_len *1.02))
    plt.ylim(0, max(max(avg_training_loss[9:]), max(avg_validation_loss[9:])) * 1.1)
    plt.legend()

    save_path = f"Visualizations/Loss_scores/{ml_models_dir.replace('ML_saved_models/', '').replace('LSTM_', '')}"
    plt.savefig(f"{save_path}_training_loss.png")

    plt.show()


def safe_nanmean(data, threshold=5):
    """Calculate the mean, ignoring NaN, if non-NaN count meets the threshold."""
    if np.count_nonzero(~np.isnan(data)) >= threshold:
        return np.nanmean(data)
    else:
        return np.nan  


def safe_nanstd(data, threshold=5):
    """Calculate the std deviation, ignoring NaN, if non-NaN count meets the threshold."""
    if np.count_nonzero(~np.isnan(data)) >= threshold:
        return np.nanstd(data)
    else:
        return np.nan  


if __name__ == "__main__":

    directories = ["LSTM_all_features_MSE", "LSTM_OHLC_MSE", "LSTM_all_features_custom_loss", "LSTM_OHLC_custom_loss" ]
    best_model_loss_scores = {"model_type": [],"model_name" : [], "val_loss" : [], "test_loss" : [], "loss_type" : [], "epochs_trained" : []}
    mean_model_loss_scores = {"model_type": [],"val_loss" : [], "test_loss" : [], "loss_type" : [], "epochs_trained" : []}
    std_model_loss_scores  = {"model_type": [],"val_loss" : [], "test_loss" : [], "loss_type" : [], "epochs_trained" : []}


    for cur_dir in directories:
        print("#############################################")
        print("\n")
        print("Model types:                 : ", cur_dir)


        cur_dir = f"ML_saved_models/{cur_dir}"
        all_metrics_df = combine_all_metrics(cur_dir)

        stats_df = aggregate_metrics(all_metrics_df)
        
        print("")
        print("stats_df:")
        print(stats_df)
        print_metrics_for_best_model(all_metrics_df, cur_dir)

        plot_training_loss(cur_dir, 'loss')
        print("#############################################")


        # Non pretty, and very stressful code, but i had to do it fast. It just sorts out the best model for each model type, along with some mean and std values.
        # Not clean code, please don't judge me.
        best_model_loss_scores["model_type"].append(cur_dir.replace("ML_saved_models/", "").replace("LSTM_", "").replace("_MSE", " ").replace("_custom_loss", " "))
        best_model_loss_scores["model_name"].append(all_metrics_df.loc[all_metrics_df["val_loss"].idxmin(), "model_name"])
        best_model_loss_scores["val_loss"].append(all_metrics_df.loc[all_metrics_df["val_loss"].idxmin(), "val_loss"])
        best_model_loss_scores["test_loss"].append(all_metrics_df.loc[all_metrics_df["val_loss"].idxmin(), "test_loss"])
        if "MSE" in cur_dir:
            best_model_loss_scores["loss_type"].append("MSE")
        else: 
            best_model_loss_scores["loss_type"].append("Custom")
        best_model_loss_scores["epochs_trained"].append(all_metrics_df.loc[all_metrics_df["val_loss"].idxmin(), "epochs_trained"])

        mean_model_loss_scores["model_type"].append(cur_dir.replace("ML_saved_models/", "").replace("LSTM_", "").replace("_MSE", " ").replace("_custom_loss", " "))
        mean_model_loss_scores["val_loss"].append(all_metrics_df["val_loss"].mean())
        mean_model_loss_scores["test_loss"].append(all_metrics_df["test_loss"].mean())
        if "MSE" in cur_dir:
            mean_model_loss_scores["loss_type"].append("MSE")
        else: 
            mean_model_loss_scores["loss_type"].append("Custom")
        mean_model_loss_scores["epochs_trained"].append(all_metrics_df["epochs_trained"].mean())

        std_model_loss_scores["model_type"].append(cur_dir.replace("ML_saved_models/", "").replace("LSTM_", "").replace("_MSE", " ").replace("_custom_loss", " "))
        std_model_loss_scores["val_loss"].append(all_metrics_df["val_loss"].std())
        std_model_loss_scores["test_loss"].append(all_metrics_df["test_loss"].std())
        if "MSE" in cur_dir:
            std_model_loss_scores["loss_type"].append("MSE")
        else: 
            std_model_loss_scores["loss_type"].append("Custom")
        std_model_loss_scores["epochs_trained"].append(all_metrics_df["epochs_trained"].std())


    best_model_loss_scores_df = pd.DataFrame(best_model_loss_scores)
    best_model_loss_scores_df = best_model_loss_scores_df[['model_type', 'loss_type', 'val_loss', 'test_loss',  'epochs_trained','model_name']]
    mean_model_loss_scores_df = pd.DataFrame(mean_model_loss_scores)
    mean_model_loss_scores_df = mean_model_loss_scores_df[['model_type', 'loss_type', 'val_loss', 'test_loss',  'epochs_trained']]
    std_model_loss_scores_df = pd.DataFrame(std_model_loss_scores)
    std_model_loss_scores_df = std_model_loss_scores_df[['model_type', 'loss_type', 'val_loss', 'test_loss',  'epochs_trained']]

    print("\n")
    print("best_model_loss_scores_df:")
    print(best_model_loss_scores_df.to_latex(index=False))
    print(best_model_loss_scores_df)
    print("\n")
    print("mean_model_loss_scores_df:")
    print(mean_model_loss_scores_df.to_latex(index=False))
    print(mean_model_loss_scores_df)
    print("\n")
    print("std_model_loss_scores_df:")
    print(std_model_loss_scores_df.to_latex(index=False))
    print(std_model_loss_scores_df)
    print("\n")


