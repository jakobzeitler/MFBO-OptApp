#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import json
from Model_maker import BuildModel, Syngas_fermentation_simulator
from utils import generate_initial_data, tensor_to_dict, set_rng_seed
from Multi_fedility_maker import Multi_fidelity_model
from MHSapi.MHSapi import MHSapiClient
import pandas as pd

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Download an OpenML dataset')
    parser.add_argument('--token', help='token', required=True, default='01ba5d69504449ce8de4faca341e')
    parser.add_argument('--base_url', help='base url', required=True, default= 'https://mhs.ngrok.app/' )
    parser.add_argument('--project_id', help='project id', required=True, default=-1)
    parser.add_argument('--opt_run_id', help='opt run id', required=True, default=-1)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--meta', help='metadata file', default='meta.json')

    args = parser.parse_args()

    # 1. Initialise API client
    client = MHSapiClient(token=args.token, base_url=args.base_url)
    projects = client.experiments_list()
    project = [p for p in projects if int(p.id) == int(args.project_id)][0]
    parameters = client.parameters_list(project)

    # 2. Download dataset
    dataset = client.experiment_data(project)
    print("Data")
    print(dataset)

    # 3. Save data
    dataset.to_csv(args.data, sep='\t')

    inputs = [p for p in parameters if p.outcome == False and p.timestamp == False] #and p.fidelity == False]
    outcome = [p for p in parameters if p.outcome == True][0]
    #fidelity = [p for p in parameters if p.outcome == False and p.timestamp == False and p.fidelity == True][0]

    #fidelity = dataset[[fidelity.parameter_text]]
    X = dataset[[i.parameter_text for i in inputs]]
    #X = pd.concat([X,fidelity])
    Y = dataset[[outcome.parameter_text]]
    train_x = torch.tensor(X.to_numpy(dtype=np.float64))
    train_obj = torch.tensor(Y.to_numpy(dtype=np.float64))
    set_rng_seed(1245)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), }

    # TODO: pull from MHS
    bounds = torch.tensor([[1, 101325, 0.1, 0.1, 0.1, 1, 1E-3, 1], [50.0, 506625, 1, 1, 1, 200, 1E-2, 1]], **tkwargs)
    # target_fidelities = {7: 1.0}
    gp_model = BuildModel()
    problem = Syngas_fermentation_simulator(gp_model, negate=False)

    # train_x, train_obj = generate_initial_data(problem, n=5)
    fidelities = torch.tensor([0.0, 1.0], **tkwargs)

    fidelity_list = tensor_to_dict(fidelities, problem.dim - 1)
    row_indices = np.array([2, 3, 4])
    fixed_cost = 0

    mf_model = Multi_fidelity_model(problem, fidelity_list, row_indices, fixed_cost, bounds, train_x, train_obj)

    data, cost = mf_model.run()
    data.to_csv(f'{project.id}_Dataset.csv')
    model = mf_model.get_model()
    final_rec = mf_model.get_recommendation(model)
    # print(data)
    # print('----')
    df_shape = data.shape

    # Access the number of rows and columns
    num_rows = df_shape[0]
    num_columns = df_shape[1]
    print(f"Input shape: {len(inputs)} and Columns: {num_columns}")
    new_sample = {}
    for col in range(num_columns - 1):
        for row in range(num_rows):
            new_sample[inputs[col].parameter_text] = data.iloc[row, col]

    # new_sample = {}
    # for i, c in data.iterrows():
    #     print([i, c])
    #     new_sample[inputs[i].parameter_text] = c

    new_sample[outcome.parameter_text] = np.nan
    new_sample["opt_run_id"] = args.opt_run_id
    new_sample = pd.DataFrame(new_sample, index=[0])
    print(new_sample)
    client.experiment_update_data(project, new_sample)

    # save metadata
    meta = {
        'project_id': args.project_id,
        'base_url':client.get_base_url()
    }

    with open(args.meta, 'w') as f:
        json.dump(meta, f)
