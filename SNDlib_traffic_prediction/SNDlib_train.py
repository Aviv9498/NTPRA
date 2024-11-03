import copy
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import wandb
from SNDlib_data import load_dataframe_from_csv, prepare_datasets
from SNDlib_Prediction_Model import GRUPredictor
from SNDlib_Prediction_Model import GNNPredictorWithDistance


def train(X_train, y_train, X_val, y_val, model, model_name, dataset, num_epochs, batch_size, optimizer, criterion, lr, weight_decay,
          l1_parameter, device, print_every=10, save_every=10, load_model=False, WANDB_TRACKING=True, distance=None, use_schedualer=True
          ):

    # ------------ WANDB Tracking----------------------#
    if WANDB_TRACKING:
        wandb.login(key="3ec39d34b7882297a057fdc2126cd037352175a4")
        wandb.init(project="SNDlib Traffic Prediction",
                   config={
                       "Model_Name": model_name,
                       "epochs": num_epochs,
                       "batch_size": batch_size,
                       "lr": lr,
                    })

    print(f"Training : {model_name} Model on : {dataset} Dataset : \n")

    # --------------- Defining Directories ----------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # base_dir = '/content/gdrive/My Drive/SNDlib/'

    # Construct the save directory relative to the base directory
    save_dir = os.path.join(base_dir, 'SNDlib_models')
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------#

    if load_model:
        model_path = os.path.join(save_dir, f'{dataset}_model_best_model.pk')
        model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)

    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_model = model
    best_val_loss = np.inf
    best_epoch = 0
    # Initializing arrays for plotting
    train_loss = []
    validation_loss = []

    # start training loop
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # len X_batch doesn't surely divide by batch size, we skip last minibatch
            if np.shape(X_batch)[0] != batch_size:
                continue

            #  just in this model the input is not raw data
            if (isinstance(model, GNNPredictorWithDistance)) and (distance is not None):
                output = model(X_batch, distance)
            else:
                output = model(X_batch)        # output = model(X_batch.unsqueeze(-1))

            loss = criterion(output, y_batch)
            # Adding L1 reg
            loss += l1_parameter * l1_reg(model=model)
            # loss = torch.sqrt_(criterion(output, y_batch.unsqueeze(-1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()

        running_loss /= (len(X_train) // batch_size)  # Average loss
        train_loss.append(running_loss)

        # Evaluating the model
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i in range(0, X_val.size(0), batch_size):
                X_batch, y_batch = X_val[i:i + batch_size], y_val[i:i + batch_size]
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # len X_batch doesn't surely divide by batch size, we skip last minibatch
                if np.shape(X_batch)[0] != batch_size:
                    continue

                #  just in this model the input is not raw data
                if (isinstance(model, GNNPredictorWithDistance)) and (distance is not None):
                    val_output = model(X_batch, distance)
                else:
                    val_output = model(X_batch)

                # running_loss += torch.sqrt_(criterion(val_output, y_batch.unsqueeze(-1))).item()
                running_loss += criterion(val_output, y_batch).item()

        running_loss /= (len(X_val) // batch_size)
        validation_loss.append(running_loss)

        if use_schedualer:
            scheduler.step(running_loss)  # Step the scheduler with the validation loss

        if WANDB_TRACKING:
            wandb.log({"Train Loss": train_loss[-1], "Val Loss": validation_loss[-1],
                       "Learning Rate": optimizer.param_groups[0]['lr']})

        if (epoch + 1) % print_every == 0:
            print(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f} , Validation Loss: {validation_loss[-1]:.4f}')

        if (epoch + 1) % save_every == 0:
            # Construct the checkpoint path, save weights every few epochs
            # checkpoint_path = os.path.join(save_dir)
            checkpoint_path = os.path.join(save_dir, f'{model_name}_model_{epoch+1}.pk')
            torch.save(model.state_dict(), checkpoint_path)

        # Check if model improved
        if validation_loss[-1] < best_val_loss:
            best_val_loss = validation_loss[-1]
            best_model_state = copy.deepcopy(model.state_dict())  # Save a deep copy of the state_dict
            best_epoch = epoch

            # save best model Everytime it changes
            # checkpoint_path = os.path.join(save_dir, f'{model_name}'))
            checkpoint_path = os.path.join(save_dir, f'{dataset}_{model_name}_model_best_model.pk')
            if best_model_state is not None:
                torch.save(best_model_state, checkpoint_path)

    # save best model
    # checkpoint_path = os.path.join(save_dir, f'{model_name}'))
    checkpoint_path = os.path.join(save_dir, f'{dataset}_{model_name}_model_best_model.pk')
    if best_model_state is not None:
        torch.save(best_model_state, checkpoint_path)

    # save last model
    checkpoint_path = os.path.join(save_dir, f'{dataset}_{model_name}_model_last_model.pk')
    torch.save(model.state_dict(), checkpoint_path)

    print(f'Finished Training {dataset} {model_name}, best model was epoch: {best_epoch}')

    if WANDB_TRACKING:
       wandb.finish()

    return np.array(train_loss), np.array(validation_loss), best_model


def l1_reg(model):

    l1_regularization = 0.0

    for param in model.parameters():
        l1_regularization += torch.sum(torch.abs(param))

    return l1_regularization


def plot_curves(train_losses, val_losses, model_name, dataset):
    plt.figure()

    plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
    plt.plot(np.arange(len(val_losses)), val_losses, label='Validation')

    plt.title(f"MSE Loss Curves")
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_prediction_loss_curves')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{dataset}_training_{model_name}.png')
    plt.savefig(save_path)
    plt.close()


def plot_MAE_bar(models, mae_values):
    # Creating figure and axis objects
    fig, ax1 = plt.subplots()

    # Plotting the bar plot for RMSE
    ax1.bar(models, mae_values, color='skyblue', alpha=0.8)
    ax1.set_ylabel('MAE', color='blue')
    # ax1.set_ylim(0, 1)  # Adjust according to your data range
    ax1.set_xlabel('Models')

    plt.title("MAE Results for different Models ")
    plt.grid(True)

    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_prediction_loss_curves')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'MAE_Results.png')
    plt.savefig(save_path)
    plt.close()


def plot_mae_mse_bar(models, mse_values, mae_values, dataset):
    # Number of models
    n_models = len(models)
    bar_width = 0.35  # Adjust bar width for better spacing
    index = np.arange(n_models)

    # Colors and hatching patterns for each model
    colors = ['limegreen', 'orange', 'blue', 'gray', 'dodgerblue', 'purple']
    hatches = ['/', '\\', '|', '-', '+', 'x']

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Adjust y-axis limits to highlight differences
    mse_ylim = (min(mse_values) * 0.95, max(mse_values) * 1.05)
    mae_ylim = (min(mae_values) * 0.95, max(mae_values) * 1.05)

    # Plotting MSE
    for i, model in enumerate(models):
        ax1.bar(index[i], mse_values[i], bar_width, color=colors[i], hatch=hatches[i], edgecolor='black', label=model)
        ax1.text(index[i], mse_values[i], f'{mse_values[i]:.4f}', ha='center', va='bottom', fontsize=9)  # Annotate bars

    ax1.set_xlabel('Models')
    ax1.set_ylabel('MSE')
    ax1.set_title('MSE Results')
    ax1.set_xticks(index)
    ax1.set_xticklabels(models)
    ax1.set_ylim(mse_ylim)  # Set tighter limits
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plotting MAE
    for i, model in enumerate(models):
        ax2.bar(index[i], mae_values[i], bar_width, color=colors[i], hatch=hatches[i], edgecolor='black', label=model)
        ax2.text(index[i], mae_values[i], f'{mae_values[i]:.4f}', ha='center', va='bottom', fontsize=9)  # Annotate bars

    ax2.set_xlabel('Models')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE Results')
    ax2.set_xticks(index)
    ax2.set_xticklabels(models)
    ax2.set_ylim(mae_ylim)  # Set tighter limits
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('MSE and MAE Results for Different Models')

    # Ensure the directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Traffic_prediction_loss_curves')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{dataset}_MSE_MAE_Results.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()


def split_test_set_into_days(X_test, y_test, rows_per_day=288):
    sequences_per_day = rows_per_day // y_test.size(1)  # Number of sequences per day
    num_days = len(X_test) // sequences_per_day

    days_X_test = X_test[:num_days * sequences_per_day].view(num_days, sequences_per_day, X_test.size(1), X_test.size(2))
    days_y_test = y_test[:num_days * sequences_per_day].view(num_days, sequences_per_day, y_test.size(1), y_test.size(2))

    return days_X_test, days_y_test


def test_model(model, X_test, y_test):
    print("Checking Accuracy : \n")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        days_X_test, days_y_test = split_test_set_into_days(X_test, y_test)
        mae_per_day = []

        for day_X, day_y in zip(days_X_test, days_y_test):
            day_predictions = []
            for X_seq in day_X:
                X_seq = X_seq.unsqueeze(0)  # Add batch dimension to make it (1, 72, 132)
                prediction = model(X_seq)
                day_predictions.append(prediction.squeeze(0))  # Remove batch dimension

            day_predictions = torch.cat(day_predictions, dim=0).view(-1)
            day_actuals = day_y.view(-1)
            mae = torch.mean(torch.abs(day_predictions - day_actuals)).item()
            mae_per_day.append(mae)

    mean_mae = np.mean(mae_per_day)

    print(f'Mean Absolute Error (MAE) over one day intervals: {mean_mae:.4f}')
    return mean_mae


def MAE_MAPE_MSE_test(model, X_test, y_test, device, distance=None):
    print("Checking Accuracy : \n")

    all_mae = []
    all_mape = []
    all_mse = []  # List to store MSE values
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in zip(X_test, y_test):
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(0)

            # Just in this model, the input is not raw data
            if (isinstance(model, GNNPredictorWithDistance)) and (distance is not None):
                prediction = model(x, distance)
            else:
                prediction = model(x)

            # Flatten predictions and targets
            prediction_flat = prediction.view(-1)
            y_flat = y.view(-1)

            # Ensure that the actual values are not zero to avoid division by zero
            non_zero_y = y_flat + (y_flat == 0).float() * 1e-8  # Add a small epsilon where y is zero

            # MAE
            mae = torch.mean(torch.abs(prediction_flat - y_flat)).item()
            all_mae.append(mae)

            # MAPE
            mape = torch.mean(torch.abs((prediction_flat - y_flat) / non_zero_y)).item() * 100
            all_mape.append(mape)

            # MSE
            mse = torch.mean((prediction_flat - y_flat) ** 2).item()
            all_mse.append(mse)

    mean_mae = np.mean(all_mae)
    mean_mape = np.mean(all_mape)
    mean_mse = np.mean(all_mse)

    print(
        f'Mean Absolute Error (MAE): {mean_mae:.4f}, Mean Absolute Percentage Error (MAPE): {mean_mape:.4f}, Mean Squared Error (MSE): {mean_mse:.4f} \n')

    return mean_mae, mean_mape, mean_mse
