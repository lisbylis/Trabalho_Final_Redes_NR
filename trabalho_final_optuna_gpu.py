import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from optuna import create_study, Trial

from classes import DataModule, MLP, cria_modelo, funcao_objetivo

NUM_TENTATIVAS = 100

objeto_de_estudo = create_study(direction="minimize")

objeto_de_estudo.optimize(funcao_objetivo, n_trials=NUM_TENTATIVAS)

melhor_trial = objeto_de_estudo.best_trial

print(f"Número do melhor trial: {melhor_trial.number}")
print(f"Parâmetros do melhor trial: {melhor_trial.params}")
