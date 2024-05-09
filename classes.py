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

TAMANHO_TESTE = 0.1
TAMANHO_VALIDACAO = 0.1
SEMENTE_ALEATORIA = 61455

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        tamanho_lote=256,
        num_trabalhadores=2
    ):
        super().__init__()

        self.tamanho_lote = tamanho_lote
        self.num_trabalhadores = num_trabalhadores

    def setup(self, stage):
        """Ocorre após o `prepare_data`. Aqui devemos alterar o estado da classe
        para adicionar as informações referentes aos conjuntos de treino, teste
        e validação. O argumento `stage` deve existir e ele indica em qual
        estágio o processo de treino está (pode ser `fit` para
        treinamento/validação e `test` para teste).

        É nesta etapa onde aplicamos transformações aos dados caso necessário.
        """
        nome_do_arquivo = "tornado.csv"
        df = pd.read_csv(nome_do_arquivo)

        features = ["mo", "slat", "slon", "len", "wid"]
        target = ["mag"]

        df = df.reindex(features + target, axis=1)
        df = df.dropna()
        
        for index, i in enumerate(df['mag']):
            if i == -9:
                df = df.drop(index, axis=0)

        indices = df.index
        indices_treino_val, indices_teste = train_test_split(
            indices, test_size=TAMANHO_TESTE, random_state=SEMENTE_ALEATORIA
        )

        df_treino_val = df.loc[indices_treino_val]
        df_teste = df.loc[indices_teste]

        indices = df_treino_val.index
        indices_treino, indices_val = train_test_split(
            indices,
            test_size=TAMANHO_TESTE,
            random_state=SEMENTE_ALEATORIA,
        )

        df_treino = df.loc[indices_treino]
        df_val = df.loc[indices_val]

        X_treino = df_treino.reindex(features, axis=1).values
        y_treino = df_treino.reindex(target, axis=1).values

        self.x_scaler = MaxAbsScaler()
        self.x_scaler.fit(X_treino)

        self.y_scaler = MaxAbsScaler()
        self.y_scaler.fit(y_treino)

        if stage == "fit":
            X_val = df_val.reindex(features, axis=1).values
            y_val = df_val.reindex(target, axis=1).values

            X_treino = self.x_scaler.transform(X_treino)
            y_treino = self.y_scaler.transform(y_treino)

            X_val = self.x_scaler.transform(X_val)
            y_val = self.y_scaler.transform(y_val)

            self.X_treino = torch.tensor(X_treino, dtype=torch.float32)
            self.y_treino = torch.tensor(y_treino, dtype=torch.float32)

            self.X_val = torch.tensor(X_val, dtype=torch.float32)
            self.y_val = torch.tensor(y_val, dtype=torch.float32)

        if stage == "test":
            X_teste = df_teste.reindex(features, axis=1).values
            y_teste = df_teste.reindex(target, axis=1).values

            X_teste = self.x_scaler.transform(X_teste)
            y_teste = self.y_scaler.transform(y_teste)

            self.X_teste = torch.tensor(X_teste, dtype=torch.float32)
            self.y_teste = torch.tensor(y_teste, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_treino, self.y_treino),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )

    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_teste, self.y_teste),
            batch_size=self.tamanho_lote,
            num_workers=self.num_trabalhadores,
        )
    
class MLP(L.LightningModule):
    def __init__(
        self, num_dados_entrada, num_neuronios, num_camadas, num_targets, ativacao, optimizer, lr
    ):
        super().__init__()

        self.num_dados_entrada = num_dados_entrada
        self.num_neuronios = num_neuronios
        self.num_camadas = num_camadas
        self.num_targets = num_targets

        camadas = []
        camadas.append(nn.Linear(num_dados_entrada,num_neuronios))
        camadas.append(ativacao)
        for _ in range(num_camadas - 1):
            camadas.append(nn.Linear(num_neuronios,num_neuronios))
            camadas.append(ativacao)
        camadas.append(nn.Linear(num_neuronios,num_targets))

        self.rede = nn.Sequential(*camadas)

        self.fun_perda = F.mse_loss

        self.perdas_treino = []
        self.perdas_val = []

        self.curva_aprendizado_treino = []
        self.curva_aprendizado_val = []

        self.optimizer = optimizer
        self.lr = lr

    def forward(self, x):
        return self.rede(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("loss", loss, prog_bar=True)
        self.perdas_treino.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("val_loss", loss, prog_bar=True)
        self.perdas_val.append(loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fun_perda(y, y_pred)

        self.log("test_loss", loss)

        return loss

    def on_train_epoch_end(self):
        # Atualiza curva de aprendizado
        perda_media = torch.stack(self.perdas_treino).mean()
        self.curva_aprendizado_treino.append(float(perda_media))
        self.perdas_treino.clear()

    def on_validation_epoch_end(self):
        # Atualiza curva de aprendizado
        perda_media = torch.stack(self.perdas_val).mean()
        self.curva_aprendizado_val.append(float(perda_media))
        self.perdas_val.clear()

    def configure_optimizers(self):
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = self.lr)
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr = self.lr)
        elif self.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr = self.lr)
        return optimizer
    
def cria_modelo(trial):
    num_neuronios = trial.suggest_int('num_neuronios', 3, 16)
    num_camadas = trial.suggest_int('num_camadas', 1, 3)
    func_ativacao = trial.suggest_categorical('ativacao', ['leaky_relu', 'relu', 'sigmoid', 'tanh'])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    funcoes_ativacao = {
        'leaky_relu': nn.LeakyReLU(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh()
    }

    ativacao = funcoes_ativacao[func_ativacao]

    modelo = MLP(
        num_dados_entrada = 5, num_neuronios = num_neuronios , num_camadas = num_camadas, 
        num_targets = 1, ativacao = ativacao, optimizer = optimizer, lr = lr
    )

    return modelo

def funcao_objetivo(trial):
    modelo = cria_modelo(trial)

    treinador = L.Trainer(max_epochs=50, accelerator="gpu")
    
    dm = DataModule()    
    
    treinador.fit(modelo, dm)

    modelo.eval()
    
    dm.setup("test")

    with torch.no_grad():
        X_true = dm.X_teste
    
        y_true = dm.y_teste
        y_true = dm.y_scaler.inverse_transform(y_true)
    
        y_pred = modelo(X_true)
        y_pred = dm.y_scaler.inverse_transform(y_pred)
    
        RMSE = mean_squared_error(y_true, y_pred, squared=False)

    return RMSE