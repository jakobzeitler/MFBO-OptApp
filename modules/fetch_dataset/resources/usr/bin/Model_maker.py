import os
from typing import Optional
import torch
from torch import Tensor
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import warnings
from botorch.test_functions.synthetic import SyntheticTestFunction
from utils import pick_continuous_number
from utils import outcome_constraint

warnings.filterwarnings('ignore')
SMOKE_TEST = os.environ.get("SMOKE_TEST")


class BuildModel:  # Low-Fidelity Simulator 3l reactor

    def __init__(self):
        self.dim = 8
        self.bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.T = 273.15  # K
        self.R = 8.314  # J/mol.K
        self.A = 16.63  # m2
        self.klco = 0.000398
        self.klh2 = 0.000593
        self.klco2 = 0.000387
        self.qcomax, self.qh2max = 1.459, 2.565  # mol/molx/h
        self.Ksco, self.Ksh2 = 0.042, 0.025  # mol/m3
        self.Ki, self.Kico = 0.246, 0.025  # mol2/m6, mol/m3
        # Initial quess for the scipy
        self.clCO_Emean = [6.99E-01, 7.82E-02, 3.10E-02, 2.02E-02, 4.74E-03]  # mM
        self.clH2_Emean = [2.93E-01, 3.69E-02, 0.01154882, 6.84E-03, 9.27E-04]  # mM
        self.clCO2_Emean = [10, 8, 5, 10, 2]  # mM
        self.qCO_Emean = [3.92E-01, 6.13E-01, 4.52E-01, 3.57E-01, 1.33E-01]  # mol/mol/h
        self.qH2_Emean = [8.82E-02, 2.62E-01, 2.02E-01, 1.59E-01, 5.99E-02]  # mol/mol/h
        self.row_indices = [2, 3, 4]

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), }

    '''
     This function calculates the kla used in equation (5) of Lars's paper.
    '''

    def kla(self, ngs, p, db):
        ugs = ngs * self.R * self.T / (self.A * p)
        epsilon_g = 4 * ugs
        kla_h2 = 6 * self.klh2 * epsilon_g / db
        kla_co = 6 * self.klco * epsilon_g / db
        kla_co2 = 6 * self.klco2 * epsilon_g / db

        return [kla_co, kla_h2, kla_co2]

    '''
     This function calculates the q of equation (6) and (7) of the paper.
    '''

    def q_rate(self, c):
        clco, clh2, clco2 = c[0], c[1], c[2]  # parameters for reaction

        qco = self.qcomax * (clco / (self.Ksco + clco + clco ** 2 / self.Ki))  # mol/molx/h
        qh2 = self.qh2max * (clh2 / (self.Ksh2 + clh2)) * (1 / (1 + clco / self.Kico))  # mol/molx/h
        qco2 = -4.0 / 6.0 * qco + 1.0 / 3.0 * qh2

        return [qco, qh2, qco2]

    '''
    There is a system of equation for ideal mixing in Eq. (6), (7), and (8) that
    gives as output the cl values. This function formulates it and going to be
    solved later with scipy
    '''

    def f_(self, c, cx, phead, yco, yh2, yco2, ngs, db):
        # cx = 5; kla_h2 = 0.1;
        clco, clh2, clco2 = c[0], c[1], c[2]
        # yco2 = 1 - yco - yco
        [kla_co, kla_h2, kla_co2] = self.kla(ngs, phead, db)

        # other parameters (as input for model) for mass transfer
        Hco, Hh2, Hco2 = 2.3e-7, 1.47e-8, 1.06e-5  # kg/m3/Pa
        rho = 998  # kg/m3
        g = 9.81  # m/s2
        H = 25  # m
        MWco, MWh2, MWco2, MWx = 28, 2, 44, 24.6  # g/mol

        # calculate solubilities
        pav = phead + 1 / 2 * (rho * g * H + phead)  # Pa
        csco = Hco * pav * yco * 1000 / MWco  # mol/m3
        csh2 = Hh2 * pav * yh2 * 1000 / MWh2  # mol/m3
        csco2 = Hco2 * pav * yco2 * 1000 / MWco2  # mol/m3 {yco2 is the remainer of 1 - yco - yh2

        qco, qh2, qco2 = self.q_rate(c)  # mol/mol/h
        rco = qco * cx / MWx * 1000 / 3600  # mol/m3/s
        rh2 = qh2 * cx / MWx * 1000 / 3600  # mol/m3/s
        rco2 = qco2 * cx / MWx * 1000 / 3600  # mol/m3/s

        return [
            kla_co * (csco - clco) - rco,
            kla_h2 * (csh2 - clh2) - rh2,
            kla_co2 * (csco2 - clco2) - rco2
        ]

    '''
    This function is used for generating the starting point of the algorithm.
    (It will further replaced with the numirical results obtained from Fig. 3
    of the paper.)
    '''

    def create_custom_tensor(self, n):
        tensor = torch.zeros(n, 7, dtype=torch.double)
        tensor[:, 0] = pick_continuous_number(1, 50)  # cX
        tensor[:, 1] = torch.randint(101325, 506626, (n,), dtype=torch.double)  # pressure P in Pa
        tensor[:, 2] = pick_continuous_number(0.01, 1)  # yCO
        tensor[:, 3] = pick_continuous_number(0.01, 1)  # yH2
        tensor[:, 4] = pick_continuous_number(0.01, 1)  # yCO2
        tensor[:, 5] = pick_continuous_number(20, 200)  # ngs
        tensor[:, 6] = pick_continuous_number(1E-3, 1E-2)  # db in m
        return tensor

    '''
    This is the function that solves the f_ function to obtain cl
    '''

    def cl_out(self, X):

        cco_ini = self.clCO_Emean[3]
        ch2_ini = self.clH2_Emean[3]
        co2_init = self.clCO2_Emean[3]
        initial_guess = np.array([cco_ini, ch2_ini, co2_init])
        Q_out = []

        tensor_array = np.array(X)

        # Reshape the tensor to a 2D matrix
        matrix = tensor_array.reshape(tensor_array.shape[0], -1)

        cx = matrix[:, 0]
        phead_ = matrix[:, 1]
        yco_ = matrix[:, 2]
        yh2_ = matrix[:, 3]
        yco2_ = matrix[:, 4]
        ngs_ = matrix[:, 5]
        db_ = matrix[:, 6]
        n_batch = matrix.shape[0]
        for n in range(n_batch):
            solution = fsolve(self.f_, initial_guess,
                              args=(
                                  cx[n], phead_[n], yco_[n], yh2_[n], yco2_[n], ngs_[n],
                                  db_[n]))

            # Extract the values of clco and clh2 from the solution
            clco_solution, clh2_solution, clco2_solution = solution[0], solution[1], solution[2]

            Q = self.q_rate([clco_solution, clh2_solution, clco2_solution])
            Q_out.append(([Q[0]]))  # .  JUST FOR THE CO

            # Q_out.append(np.array([Q[0], Q[1]]))
            cl_out = np.array([clco_solution, clh2_solution])

        return torch.tensor(Q_out, **self.tkwargs)

    def get_high_fidelity_data(self):
        self.df = pd.read_csv('Data/Dataset.csv')
        return self.df

    # def check_availability(self, X):

    #     r_tgt = []
    #     self.get_high_fidelity_data()
    #     # indices_of_last_entry_1 = torch.nonzero(X[:, -1] == 1).squeeze()

    #     self.df = self.df.dropna(how='all')
    #     high_fed = self.df.iloc[:, :8].values
    #     R, C = high_fed.shape

    #     try:
    #         R_input, C_input = X.shape
    #     except:
    #         X = np.expand_dims(X, axis=0)
    #         X = outcome_constraint(X, self.row_indices)
    #         R_input, C_input =  X.shape # For the case when row_hf has just one dim
    #     obj = []
    #     for r in range(R):
    #         for r_input in range(R_input):
    #             if np.all(np.abs(high_fed[r, :] - X[r_input, :]) <= 2):
    #                 obj.append(self.df.iloc[r_input, -1])
    #                 r_tgt.append([r, r_input])
        
    #     if len(r_tgt) == 0:
    #         new_array = np.column_stack((X, np.full(X.shape[0], 0.8)))
    #         new_array = pd.DataFrame(new_array, columns=self.df.columns)
    #         self.df = pd.concat([self.df, new_array], ignore_index=True)
    #         self.df.to_csv('Data/Dataset.csv', index=False)
    #         # raise ValueError(f'Please update the dataset with the given values{X}. The objective is set to { 0.8}.'
    #         #                  f' Change it with the actual result and rerun the code.')

    #     obj = np.array(obj, dtype=np.float64)
    #     obj_torch = torch.tensor(obj, dtype=torch.float64)



    #     return obj_torch.unsqueeze(-1)

    def check_availability(self, X):
        # new_array = np.column_stack((X, np.full(X.shape[0], 0.8)))


        r_tgt = []
        self.get_high_fidelity_data()
        # indices_of_last_entry_1 = torch.nonzero(X[:, -1] == 1).squeeze()

        self.df = self.df.dropna(how='all')
        high_fed = self.df.iloc[:, :8].values
        R, C = high_fed.shape

        try:
            R_input, C_input = X.shape
        except:
            X = np.expand_dims(X, axis=0)
            X = outcome_constraint(X, self.row_indices)
            R_input, C_input =  X.shape # For the case when row_hf has just one dim
        obj = []
        for r in range(R):
            for r_input in range(R_input):
                if np.all(np.abs(high_fed[r, :] - X[r_input, :]) <= 2):
                    obj.append(self.df.iloc[r_input, -1])
                    r_tgt.append([r, r_input])
        
        if len(r_tgt) == 0:
            new_array = np.column_stack((X, np.full(X.shape[0], 0.8)))
            new_array = pd.DataFrame(new_array, columns=self.df.columns)
            self.df = pd.concat([self.df, new_array], ignore_index=True)
            self.df.to_csv('Data/Dataset.csv', index=False)
            # raise ValueError(f'Please update the dataset with the given values{X}. The objective is set to { 0.8}.'
            #                  f' Change it with the actual result and rerun the code.')

        obj = np.array(obj, dtype=np.float64)
        obj_torch = torch.tensor(obj, dtype=torch.float64)



        return obj_torch.unsqueeze(-1)

    def combine_tensors_with_indices(self, tensorA, A, tensorB, B):
        """
        Create a new tensor containing elements from tensorA at indices A and elements from tensorB at indices B.

        Args:
            tensorA (torch.Tensor or None): The first input tensor or None.
            A (list or torch.Tensor or None): The indices to select from tensorA or None.
            tensorB (torch.Tensor): The second input tensor.
            B (list or torch.Tensor): The indices to select from tensorB.

        Returns:
            torch.Tensor: The new tensor containing the combined elements.
        """
        if tensorA is None or A is None:
            # If either tensorA or A is None, work with tensorB and B directly
            tensorC = torch.empty(max(B) + 1, dtype=tensorB.dtype)
            tensorC[B] = tensorB.squeeze()
        else:
            tensorC = torch.empty(max(max(A), max(B)) + 1, dtype=tensorA.dtype)
            tensorC[A] = tensorA.squeeze()
            tensorC[B] = tensorB.squeeze()

        return tensorC


class Syngas_fermentation_simulator(SyntheticTestFunction):
    dim = 8
    _bounds = [(0.0, 1.0) for _ in range(dim)]

    def __init__(self, gp_model, noise_std: Optional[float] = None, negate: bool = False) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.gp_model = gp_model

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        tensorC = torch.empty(X.size()[0], 1, dtype=X.dtype)
        indices_of_last_entry_1 = torch.nonzero(X[:, -1] == 1).squeeze()
        row_hf = X[indices_of_last_entry_1].numpy()
        high_fidelity_obj = None
        if indices_of_last_entry_1.numel() > 0:
            # high_fidelity_obj = self.gp_model.check_availability(row_hf)
            high_fidelity_obj = 2
            tensorC[indices_of_last_entry_1.numpy()] = X[indices_of_last_entry_1.numpy(), 0].unsqueeze(-1)* high_fidelity_obj

        zero_rows = (X[:, -1] == 0).nonzero(as_tuple=True)[0]

        obj_lf = None
        if torch.numel(zero_rows) != 0:
            obj_lf = self.gp_model.cl_out(X[zero_rows, :7])
            tensorC[zero_rows.numpy()] = X[zero_rows.numpy(), 0].unsqueeze(-1)* obj_lf

        # H = (self.gp_model.cl_out(X[..., :7].unsqueeze(1)))

        return tensorC
