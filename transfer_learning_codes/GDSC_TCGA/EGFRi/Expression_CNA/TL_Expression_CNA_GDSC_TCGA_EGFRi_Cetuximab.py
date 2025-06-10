# import libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import mygene
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold
import time


# start time
start = time.time()

# define directories
dataset_dir = "/arf/home/salpsoy/Thesis_Work/Datasets/"
DEGs_dir = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/EGFRi/"
cell_line_dir = "/arf/home/salpsoy/Thesis_Work/Supplementary_Files/GDSC/"
save_models_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/EGFRi/save_models/GDSC_TCGA/Expression_CNA/"
save_results_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/EGFRi/save_models/GDSC_TCGA/Expression_CNA/AUC and Cost Plots/"


# define maximum iteration and set random seed
max_iter = 50
torch.manual_seed(42)
random.seed(42)


# change directory to read DEGs
os.chdir(DEGs_dir)


# read diferentially expressed genes common in EGFR signaling inhibitors
DEGs_filtered_data = pd.read_excel("EGFRi_Differentially_Expressed_Genes (EnsemblID).xlsx",
                                   sheet_name="Common DEGs")



# get Entrez IDs from gene symbols
mg = mygene.MyGeneInfo()
DEGs_entrez_id = mg.querymany(DEGs_filtered_data["Gene Symbol"],
                              species="human",
                              scopes="symbol",
                              field="entrezgene",
                              as_dataframe=True)["entrezgene"]
DEGs_entrez_id.dropna(inplace=True)


# change directory to read multi-omics datasets
os.chdir(dataset_dir)


# read GDSC expression dataset (EGFRi)
GDSCE = pd.read_csv("GDSC_exprs.z.EGFRi.tsv",
                    sep="\t",
                    index_col=0,
                    decimal=",")
GDSCE = pd.DataFrame.transpose(GDSCE)


# read GDSC CNA dataset (EGFRi)
GDSCC = pd.read_csv("GDSC_CNA.EGFRi.tsv",
                    sep="\t", index_col=0, decimal=".")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)
GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]


# read GDSC response dataset (EGFRi) and binarize 
GDSCR = pd.read_csv("GDSC_response.EGFRi.tsv",
                    sep="\t",
                    index_col=0,
                    decimal=",")
GDSCR.dropna(inplace=True)
GDSCR.rename(mapper=str, axis='index', inplace=True)
d = {"R": 0, "S": 1}
GDSCR["response"] = GDSCR.loc[:, "response"].apply(lambda x: d[x])


# Read TCGA expression dataset (Cetuximab)
TCGAEcetuximab = pd.read_csv("TCGA_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                      sep="\t", index_col=0, decimal=",")
TCGAEcetuximab = pd.DataFrame.transpose(TCGAEcetuximab)
# TCGAEcetuximab.head(3)


# Read TCGA CNA dataset (Cetuximab)
TCGACcetuximab = pd.read_csv("TCGA_CNA.Cetuximab.tsv",
                      sep="\t", index_col=0, decimal=",")
TCGACcetuximab.drop_duplicates(keep='last')
TCGACcetuximab = pd.DataFrame.transpose(TCGACcetuximab)
TCGACcetuximab = TCGACcetuximab.loc[:, ~TCGACcetuximab.columns.duplicated()]
# TCGACcetuximab.head(3)


# variance threshold for GDSC expression dataset (EGFRi)
selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]


# fill NA values and binarize GDSC CNA dataset (EGFRi)
GDSCC = GDSCC.fillna(0)
GDSCC[GDSCC != 0.0] = 1


# select shared genes between GDSC and TCGA datasets
ls = GDSCE.columns.intersection(GDSCC.columns)
ls = ls.intersection(TCGAEcetuximab.columns)
ls = ls.intersection(TCGACcetuximab.columns)
ls = pd.unique(ls)


# select shared samples between GDSC expression, CNA, and response datasets (EGFRi)
ls2 = GDSCE.index.intersection(GDSCC.index)
ls2 = ls2.intersection(GDSCR.index)


# subset shared genes and samples in GDSC expression, CNA, and response datasets (EGFRi)
GDSCE = GDSCE.loc[ls2, ls]
GDSCC = GDSCC.loc[ls2, ls]
GDSCR = GDSCR.loc[ls2, :]


# select shared samples between TCGA expression and CNA datasets (Cetuximab)
ls3 = TCGAEcetuximab.index.intersection(TCGACcetuximab.index)


# subset shared samples and genes in TCGA expression and CNA datasets (Cetuximab)
TCGAEcetuximab = TCGAEcetuximab.loc[ls3, ls]
TCGACcetuximab = TCGACcetuximab.loc[ls3, ls]


# assign GDSC expression, CNA, and response datasets (EGFRi) to new variables
exprs_z = GDSCE
cna = GDSCC
responses = GDSCR


# select drugs in GDSC response dataset (EGFRi)
drugs = set(responses["drug"].values)
print("Drugs:", drugs)


# subset GDSC expression and CNA datasets (EGFRi) as to drugs
expression_zscores = []
CNA = []
for drug in drugs:
    samples = responses.loc[responses["drug"] == drug, :].index.values
    e_z = exprs_z.loc[samples, :]
    c = cna.loc[samples, :]
    expression_zscores.append(e_z)
    CNA.append(c)

GDSCEv2 = pd.concat(expression_zscores, axis=0)
GDSCCv2 = pd.concat(CNA, axis=0)
GDSCRv2 = responses


# filter DEGs from all genes in GDSC expression dataset (EGFRi) 
ls4 = list(set(GDSCE.columns).intersection(set(DEGs_entrez_id.astype(int))))

# filter shared samples between the subsetted GDSC expression and CNA datasets (EGFRi) 
ls5 = GDSCEv2.index.intersection(GDSCCv2.index)


# subset shared genes and samples in the subsetted GDSC expression, CNA, and response datasets (EGFRi) 
GDSCEv2 = GDSCEv2.loc[ls5, ls4]
GDSCCv2 = GDSCCv2.loc[ls5, ls4]
GDSCRv2 = GDSCRv2.loc[ls5, :]
responses.index = responses.index.values + "_" + responses["drug"].values

print(f"GDSC # of common samples and genes (Expression): {GDSCEv2.shape}")
print(f"GDSC # of common samples and genes (CNA): {GDSCCv2.shape}")
print(f"GDSC # of common samples (Response): {GDSCRv2.shape[0]}\n")

print(f"TCGA # of common samples and genes for Cetuximab (Expression): {TCGAEcetuximab.shape}")
print(f"TCGA # of common samples and genes for Cetuximab (CNA): {TCGACcetuximab.shape}\n")


# assign GDSC response dataset values (EGFRi) 
Y = GDSCRv2['response'].values


# define hyperparameters for deep neural network
ls_mb_size = [8, 16, 32, 64]    # mini-batch size
ls_h_dim = [1024, 256, 128, 512, 64, 32]   # neuron size
ls_lr = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]   # learning rate
ls_epoch = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]   # epoch size
ls_rate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # dropout rate
ls_wd = [0.01, 0.001, 0.1, 0.0001]    # weight decay


# use 5-fold cross-validation 
skf = StratifiedKFold(n_splits=5)


# train deep neural network and make predictions
for iters in range(max_iter):
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm1 = random.choice(ls_h_dim)
    hdm2 = random.choice(ls_h_dim)
    lre = random.choice(ls_lr)
    lrc = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate1 = random.choice(ls_rate)
    rate2 = random.choice(ls_rate)
    rate3 = random.choice(ls_rate)
    wd = random.choice(ls_wd)

    print(
        f'\nmb_size = {mbs},  h_dim[1,2] = ({hdm1},{hdm2}), lr[E, C] = ({lre}, {lrc}), epoch = {epch}, rate[1,2,3] = ({rate1},{rate2},{rate3}), wd = {wd}, lrCL = {lrCL}\n')

    for train_index, test_index in skf.split(GDSCEv2.values, Y):
        k = k + 1

        X_trainE = GDSCEv2.values[train_index, :]
        X_testE = GDSCEv2.values[test_index, :]
        X_trainC = GDSCCv2.values[train_index, :]
        X_testC = GDSCCv2.values[test_index, :]
        y_trainE = Y[train_index]
        y_testE = Y[test_index]

        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)

        X_trainC = np.nan_to_num(X_trainC)
        X_testC = np.nan_to_num(X_testC)

        TX_testE = torch.FloatTensor(X_testE)
        TX_testC = torch.FloatTensor(X_testC)
        ty_testE = torch.FloatTensor(y_testE.astype(int))

        # Train
        class_sample_count = np.array(
            [len(np.where(y_trainE == t)[0]) for t in np.unique(y_trainE)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_trainE])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type(
            'torch.DoubleTensor'), len(samples_weight), replacement=True)

        mb_size = mbs

        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE),
                                                      torch.FloatTensor(X_trainC), torch.FloatTensor(y_trainE.astype(int)))

        trainLoader = torch.utils.data.DataLoader(
            dataset=trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler=sampler)

        n_sampE, IE_dim = X_trainE.shape
        n_sampC, IC_dim = X_trainC.shape

        h_dim1 = hdm1
        h_dim2 = hdm2
        Z_in = h_dim1 + h_dim2
        lrE = lre
        lrC = lrc
        epoch = epch

        costtr = []
        auctr = []
        costts = []
        aucts = []

        class AEE(nn.Module):
            def __init__(self):
                super(AEE, self).__init__()
                self.EnE = torch.nn.Sequential(
                    nn.Linear(IE_dim, h_dim1),
                    nn.BatchNorm1d(h_dim1),
                    nn.ReLU(),
                    nn.Dropout(rate1))

            def forward(self, x):
                output = self.EnE(x)
                return output

        class AEC(nn.Module):
            def __init__(self):
                super(AEC, self).__init__()
                self.EnC = torch.nn.Sequential(
                    nn.Linear(IC_dim, h_dim2),
                    nn.BatchNorm1d(h_dim2),
                    nn.ReLU(),
                    nn.Dropout(rate2))

            def forward(self, x):
                output = self.EnC(x)
                return output

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.FC = torch.nn.Sequential(
                    nn.Linear(Z_in, 1),
                    nn.Dropout(rate3),
                    nn.Sigmoid())

            def forward(self, x):
                return self.FC(x)

        torch.cuda.manual_seed_all(42)

        AutoencoderE = AEE()
        AutoencoderC = AEC()

        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
        solverC = optim.Adagrad(AutoencoderC.parameters(), lr=lrC)

        Clas = Classifier()
        SolverClass = optim.Adagrad(
            Clas.parameters(), lr=lrCL, weight_decay=wd)
        C_loss = torch.nn.BCELoss()

        for it in range(epoch):
            epoch_cost4 = 0
            epoch_cost3 = []
            num_minibatches = int(n_sampE / mb_size)

            for i, (dataE, dataC, target) in enumerate(trainLoader):
                flag = 0
                AutoencoderE.train()
                AutoencoderC.train()
                Clas.train()

                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    ZEX = AutoencoderE(dataE)
                    ZCX = AutoencoderC(dataC)

                    ZT = torch.cat((ZEX, ZCX), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    Pred = Clas(ZT)

                    loss = C_loss(Pred, target.view(-1, 1))

                    y_true = target.view(-1, 1)
                    y_pred = Pred
                    AUC = roc_auc_score(
                        y_true.detach().numpy(), y_pred.detach().numpy())

                    solverE.zero_grad()
                    solverC.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    solverC.step()
                    SolverClass.step()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                    epoch_cost3.append(AUC)
                    flag = 1

            if flag == 1:
                costtr.append(torch.mean(epoch_cost4))
                auctr.append(np.mean(epoch_cost3))

            with torch.no_grad():

                AutoencoderE.eval()
                AutoencoderC.eval()
                Clas.eval()

                ZET = AutoencoderE(TX_testE)
                ZCT = AutoencoderC(TX_testC)

                ZTT = torch.cat((ZET, ZCT), 1)
                ZTT = F.normalize(ZTT, p=2, dim=0)
                PredT = Clas(ZTT)

                lossT = C_loss(PredT, ty_testE.view(-1, 1))

                print('Fold: {}; Max. iter-{}; Iter-{}; Training loss: {:.4}; Test loss: {:.4}'.format(k,
                                                                                                       iters + 1, it + 1, loss, lossT))

                y_truet = ty_testE.view(-1, 1)
                y_predt = PredT
                AUCt = roc_auc_score(
                    y_truet.detach().numpy(), y_predt.detach().numpy())

                costts.append(lossT)
                aucts.append(AUCt)

        costtr = [t.cpu().detach().numpy() for t in costtr]
        costts = [t.cpu().detach().numpy() for t in costts]

        os.makedirs(save_results_to, exist_ok=True)

        plt.plot(np.squeeze(costtr), '-r', np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2] = ({},{}), \nlr[E,C] = ({}, {}), epoch = {}, rate[1,2,3] = ({},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, lre, lrc,
                   epch, rate1, rate2, rate3, wd, lrCL)

        plt.suptitle(title, fontsize=8,  fontweight='bold')
        plt.savefig(save_results_to + title + '.png', dpi=150)
        plt.close()

        plt.plot(np.squeeze(auctr), '-r', np.squeeze(aucts), '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        title = 'AUC Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2] = ({},{}), \nlr[E, C] = ({}, {}), epoch = {}, rate[1,2,3] = ({},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, lre, lrc,
                   epch, rate1, rate2, rate3, wd, lrCL)

        plt.suptitle(title, fontsize=8,  fontweight='bold')
        plt.savefig(save_results_to + title + '.png', dpi=150)
        plt.close()


# save models
os.makedirs(save_models_to, exist_ok=True)
os.chdir(save_models_to)
torch.save(
    AutoencoderE, "Exprs_Cetuximab_GDSC_TCGA_EGFRi_Expression_CNA.pt")
torch.save(
    AutoencoderC, "CNA_Cetuximab_GDSC_TCGA_EGFRi_Expression_CNA.pt")
torch.save(Clas, 'Class_Cetuximab_GDSC_TCGA_EGFRi_Expression_CNA.pt')


# end time
end = time.time()
print(f"\nElapsed time: {(end - start)/3600} hours")
