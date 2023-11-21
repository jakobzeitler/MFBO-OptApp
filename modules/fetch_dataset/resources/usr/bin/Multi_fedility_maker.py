from botorch.models import model
import os
import torch
import warnings
from botorch import fit_gpytorch_mll
# from botorch.models.cost import AffineFidelityCostModel
from StaticCostModel import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf_mixed
from utils import outcome_constraint
import numpy as np
import pandas as pd
from utils import generate_initial_data
from Model_maker import BuildModel, Syngas_fermentation_simulator
warnings.filterwarnings('ignore')
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class Multi_fidelity_model:  # Low-Fidelity Simulator 3l reactor

    def __init__(self, problem, fidelity_list, row_indices,
                 fixed_cost, bounds, train_x, train_obj):

        self.mll = None
        self.model = None
        self.NUM_RESTARTS = 20 if not SMOKE_TEST else 2
        self.RAW_SAMPLES = 128 if not SMOKE_TEST else 4

        self.BATCH_SIZE = 3


        self.row_indices = row_indices
        self.fidelity_index = problem.dim
        self.fidelity_list = fidelity_list
        self.fixed_cost = fixed_cost
        self.target_fidelities = self.fidelity_list[-1]
        self.problem = problem
        self.gp_model = BuildModel()
        self.problem = Syngas_fermentation_simulator(self.gp_model, negate=False)
        # self.train_x, self.train_obj = generate_initial_data(self.problem, n=5)
        self.bounds = bounds
        self.train_x = train_x
        self.train_obj = train_obj


        self.df = pd.DataFrame(columns=['new_x', 'new_obj', 'cost'])
        self.data_list = []

        self.cumulative_cost = 0

        self.cost_model = AffineFidelityCostModel(fidelity_weights=self.fidelity_list[-1], fixed_cost=self.fixed_cost)
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

    def initialize_model(self):
        # define a surrogate model suited for a "training data"-like fidelity parameter
        # in dimension 6, as in [2]
        model = SingleTaskMultiFidelityGP(
            self.train_x, self.train_obj, outcome_transform=Standardize(m=1),
            data_fidelity=self.problem.dim - 1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        self.model = model
        self.mll = mll
        return mll, model

    def project(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)

    def get_mfkg(self, model):

        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=self.fidelity_index,
            columns=[self.fidelity_index - 1],
            values=[1],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=20 if not SMOKE_TEST else 2,
            raw_samples=1024 if not SMOKE_TEST else 4,
            options={"batch_limit": 10, "maxiter": 200},
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=128 if not SMOKE_TEST else 2,
            current_value=current_value,
            cost_aware_utility=self.cost_aware_utility,
            project=self.project,
        )

    torch.set_printoptions(precision=3, sci_mode=False)

    def optimize_mfkg_and_get_observation(self, mfkg_acqf):
        """Optimizes MFKG and returns a new candidate, observation, and cost."""

        # generate new candidates
        candidates, _ = optimize_acqf_mixed(
            acq_function=mfkg_acqf,
            bounds=self.bounds,
            fixed_features_list=self.fidelity_list,
            # If we have more than one feedility, it should be added to this dictionary. like: fixed_features_list=[ {7: 1.0}, {7: 0.5}] for two fedility case.
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            # batch_initial_conditions=X_init,
            options={"batch_limit": 7, "maxiter": 200},
        )

        # observe new values

        candidates = outcome_constraint(candidates, self.row_indices)
        cost = self.cost_model(candidates).sum()
        new_x = candidates.detach()
        new_obj = self.problem(new_x)
        # new_obj = self.problem(new_x).unsqueeze(-1)
        # print(f"candidates:\n{new_x}\n")
        # print(f"observations:\n{new_obj}\n\n")
        return new_x, new_obj, cost

    def get_recommendation(self, model):
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=self.fidelity_index,
            columns=[self.fidelity_index - 1],
            values=[1],
        )

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.bounds[:, :-1],
            q=1,
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        final_rec = rec_acqf._construct_X_full(final_rec)

        final_rec = outcome_constraint(final_rec, self.row_indices)
        objective_value = self.problem(final_rec)
        cost = self.cost_model(final_rec)
        # print(f"recommended point:\n{final_rec}\n\nobjective value:\n{objective_value}")
        return final_rec, objective_value, cost


    def run(self):
        cumulative_cost = 0.0
        N_ITER = 1 if not SMOKE_TEST else 1
        # train_x, train_obj = generate_initial_data(self.problem, n=5)
        for _ in range(N_ITER):
            self.mll, self.model = self.initialize_model()
            try:
                fit_gpytorch_mll(self.mll)
            except:
                print()
            mfkg_acqf = self.get_mfkg(self.model)
            new_x, new_obj, cost = self.optimize_mfkg_and_get_observation(mfkg_acqf)
            new_x_np = new_x.numpy()
            new_obj_np = new_obj.numpy()
            cost_np = cost.numpy()

            self.data_list.append({'new_x': new_x_np, 'new_obj': new_obj_np, 'cost': cost_np})
            self.train_x = torch.cat([self.train_x, new_x])
            self.train_obj = torch.cat([self.train_obj, new_obj])
            samples = torch.hstack((self.train_x, self.train_obj))
            samples_np = samples.numpy()
            samples_df = pd.DataFrame(samples_np)
            self.train_x = torch.cat([self.train_x, new_x])
            try:
                self.train_obj = torch.cat([self.train_obj, new_obj])
            except:
                self.train_obj = torch.cat([self.train_obj, new_obj.unsqueeze(-1)])
            self.cumulative_cost += cost

        return samples_df, self.cumulative_cost


    def get_model(self):
        return self.model
