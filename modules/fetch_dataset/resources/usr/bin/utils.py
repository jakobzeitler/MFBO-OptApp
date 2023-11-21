import torch
from torch import Tensor
import random
import numpy as np
global tkwargs
import pandas as pd
import os
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),}

def set_rng_seed(rng_seed: int) -> None:
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


# This function is used for initialising the input values
def pick_continuous_number(start, end, dtype=torch.double):
    number = (end - start) * torch.rand(1, dtype=dtype) + start
    return number.item()



# Since, sum(yCO, yH2, yCO2) = 1 & yCO2 = 1 - yCO - yH2, This function makes the
# Vcalues in the desired range.
def outcome_constraint(tensor, col_index) -> Tensor:
    if col_index is None:
        return tensor
    else:
        for row_indices in range(tensor.shape[0]):
            norm_f = (tensor[row_indices, col_index[0]]) + (tensor[row_indices, col_index[1]]) + (
            tensor[row_indices, col_index[2]])
            x = (tensor[row_indices, col_index[0]]) / norm_f
            y = (tensor[row_indices, col_index[1]]) / norm_f
            z = 1 - x - y
            tensor[row_indices, col_index] = torch.tensor([x, y, z])
        return tensor


def tensor_to_dict(input_tensor, scalar):
    output_list = [{scalar: value.item()} for value in input_tensor]
    return output_list


def generate_initial_data(problem, n=16):
        
    if os.path.exists('Data/Dataset.csv'):
        # File exists, you can proceed to load the dataset
        # Add your code to load the dataset here
        # If you're using pandas, you can use pd.read_csv(file_path) to load the CSV file
        df = pd.read_csv('Data/Dataset.csv', header=None, skiprows=1)
        file_found = True
    else:
        # File doesn't exist
        file_found = False

        # Check if the DataFrame is empty
        if os.path.exists('Data/Dataset.csv'):
            if not df.empty and not df.isna().any().any():
                # Select all elements except the last one for train_x
                train_x_full = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float64)

            # Select the last column for train_obj
            train_obj = torch.tensor(df.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float64)

        else:
            fidelities = torch.tensor([0.0, 1.0], **tkwargs)
            train_x = problem.gp_model.create_custom_tensor(n)
            row_indices = [2, 3, 4]  # index of the yco, yh2 and yco2 to normalize to one and yco2 = 1 - yco - yh2
            train_x = outcome_constraint(train_x, row_indices)
            train_f = fidelities[torch.randint(2, (n, 1))]
            train_x_full = torch.cat((train_x, train_f), dim=1)
            train_obj = problem(train_x_full)  # add output dimension
        return train_x_full, train_obj
