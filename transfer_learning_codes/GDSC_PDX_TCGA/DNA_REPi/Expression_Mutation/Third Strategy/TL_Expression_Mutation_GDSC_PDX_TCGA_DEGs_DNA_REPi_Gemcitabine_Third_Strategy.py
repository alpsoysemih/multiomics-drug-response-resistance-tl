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
import seaborn as sns
import sklearn.preprocessing as sk
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


# define directory paths
dataset_dir = "/arf/home/salpsoy/Thesis_Work/Datasets/"
DEGs_dir = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/DNA_REPi/"
cell_line_dir = "/arf/home/salpsoy/Thesis_Work/Supplementary_Files/GDSC/"
save_models_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/DNA_REPi/save_models/GDSC_PDX_TCGA_Third_Strategy/Expression_Mutation/"
save_results_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/DNA_REPi/save_models/GDSC_PDX_TCGA_Third_Strategy/Expression_Mutation/Cost and AUC Plots/"


# define maximum iteration and set random seed
max_iter = 50
torch.manual_seed(42)
random.seed(42)


# change directory to read cell line details
os.chdir(cell_line_dir)


# read cell line details
GDSC_cell_line_details = pd.read_excel(
    "GDSC_Cell_Lines_Details.xlsx", keep_default_na=False)
GDSC_cell_line_details.set_index("COSMIC identifier", inplace=True)
GDSC_cell_line_details = GDSC_cell_line_details.iloc[:-1, ]
GDSC_cell_line_details.index = GDSC_cell_line_details.index.astype(str)


# change directory to read DEGs
os.chdir(DEGs_dir)


# read diferentially expressed genes common in DNA replication inhibitors
DEGs_filtered_data = pd.read_excel("DNA_REPi_Differentially_Expressed_Genes (EnsemblID).xlsx",
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


# read GDSC expression dataset (DNA_REPi)
# remove samples with shared cancer types from training dataset
GDSCE = pd.read_csv("GDSC_exprs.z.DNA_REPi.tsv",
                    sep="\t",
                    index_col=0,
                    decimal=",")
GDSCE = pd.DataFrame.transpose(GDSCE)
GDSCE = pd.merge(GDSC_cell_line_details,
                 GDSCE,
                 left_index=True,
                 right_index=True)
filter = (GDSCE["GDSC\nTissue descriptor 1"] != "pancreas") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "PAAD") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "SARC") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "LIHC") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "PCPG") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "LUSC")
GDSCE = GDSCE.loc[filter, ]
GDSCE.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCE.index = GDSCE.index.astype(int)


# read GDSC mutation dataset (DNA_REPi)
# remove samples with shared cancer types from training dataset
GDSCM = pd.read_csv("GDSC_mutations.DNA_REPi.tsv",
                    sep="\t", index_col=0, decimal=".")
GDSCM.drop_duplicates(keep='last')
GDSCM = pd.DataFrame.transpose(GDSCM)
GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]
GDSCM = pd.merge(GDSC_cell_line_details,
                 GDSCM,
                 left_index=True,
                 right_index=True)
filter = (GDSCM["GDSC\nTissue descriptor 1"] != "pancreas") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "PAAD") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "SARC") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "LIHC") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "PCPG") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "LUSC")
GDSCM = GDSCM.loc[filter, ]
GDSCM.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCM.index = GDSCM.index.astype(int)


# read GDSC response dataset (DNA_REPi) and binarize
# remove samples with shared cancer types from training dataset
GDSCR = pd.read_csv("GDSC_response.DNA_REPi.tsv",
                    sep="\t",
                    index_col=0,
                    decimal=",")
GDSCR.dropna(inplace=True)
GDSCR.rename(mapper=str, axis='index', inplace=True)
d = {"R": 0, "S": 1}
GDSCR["response"] = GDSCR.loc[:, "response"].apply(lambda x: d[x])
GDSCR = pd.merge(GDSC_cell_line_details,
                 GDSCR,
                 left_index=True,
                 right_index=True)
filter = (GDSCR["GDSC\nTissue descriptor 1"] != "pancreas") & \
         (GDSCR["GDSC\nTissue descriptor 1"] != "large_intestine") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "PAAD") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "STAD") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "READ") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "COREAD") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "ESCA")
GDSCR = GDSCR.loc[filter, ]
GDSCR.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCR.index = GDSCR.index.astype(int)


# read PDX expression dataset (Gemcitabine)
PDXEgemcitabine = pd.read_csv("PDX_exprs.Gemcitabine.eb_with.GDSC_exprs.Gemcitabine.tsv",
                      sep="\t", index_col=0, decimal=",")
PDXEgemcitabine = pd.DataFrame.transpose(PDXEgemcitabine)
PDXEgemcitabine.head(3)

# read TCGA expression dataset (Gemcitabine)
TCGAEgemcitabine = pd.read_csv("TCGA_exprs.Gemcitabine.eb_with.GDSC_exprs.Gemcitabine.tsv",
                       sep="\t", index_col=0, decimal=",")
TCGAEgemcitabine = pd.DataFrame.transpose(TCGAEgemcitabine)
TCGAEgemcitabine.head(3)


# read PDX mutation dataset (Gemcitabine)
PDXMgemcitabine = pd.read_csv("PDX_mutations.Gemcitabine.tsv",
                      sep="\t", index_col=0, decimal=",")
PDXMgemcitabine.drop_duplicates(keep='last')
PDXMgemcitabine = pd.DataFrame.transpose(PDXMgemcitabine)
PDXMgemcitabine = PDXMgemcitabine.loc[:, ~PDXMgemcitabine.columns.duplicated()]

# read TCGA mutation dataset (Gemcitabine)
TCGAMgemcitabine = pd.read_csv("TCGA_mutations.Gemcitabine.tsv",
                       sep="\t", index_col=0, decimal=",")
TCGAMgemcitabine.drop_duplicates(keep='last')
TCGAMgemcitabine = pd.DataFrame.transpose(TCGAMgemcitabine)
TCGAMgemcitabine = TCGAMgemcitabine.loc[:, ~TCGAMgemcitabine.columns.duplicated()]


# variance threshold for GDSC expression dataset (DNA_REPi)
selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]


# fill NA values and binarize GDSC mutation dataset (DNA_REPi)
GDSCM = GDSCM.fillna(0)
GDSCM[GDSCM != 0.0] = 1


# select shared genes between GDSC, PDX, and TCGA datasets
ls = GDSCE.columns.intersection(GDSCM.columns)
ls = ls.intersection(PDXEgemcitabine.columns)
ls = ls.intersection(PDXMgemcitabine.columns)
ls = ls.intersection(TCGAEgemcitabine.columns)
ls = ls.intersection(TCGAMgemcitabine.columns)
ls = pd.unique(ls)


# select shared samples between GDSC expression, mutation, and response datasets (DNA_REPi)
ls2 = GDSCE.index.intersection(GDSCM.index)
ls2 = ls2.intersection(GDSCR.index)


# subset shared genes and samples in GDSC expression, mutation, and response datasets (DNA_REPi)
GDSCE = GDSCE.loc[ls2, ls]
GDSCM = GDSCM.loc[ls2, ls]
GDSCR = GDSCR.loc[ls2, :]


# select shared samples between PDX expression and mutation datasets (Gemcitabine)
ls3 = PDXEgemcitabine.index.intersection(PDXMgemcitabine.index)


# subset shared samples and genes in PDX expression and mutation datasets (Gemcitabine)
PDXEgemcitabine = PDXEgemcitabine.loc[ls3, ls]
PDXMgemcitabine = PDXMgemcitabine.loc[ls3, ls]


# select shared samples between TCGA expression and mutation datasets (Gemcitabine)
ls4 = TCGAEgemcitabine.index.intersection(TCGAMgemcitabine.index)


# subset shared samples and genes in TCGA expression and mutation datasets (Gemcitabine)
TCGAEgemcitabine = TCGAEgemcitabine.loc[ls4, ls]
TCGAMgemcitabine = TCGAMgemcitabine.loc[ls4, ls]


# assign GDSC expression, mutation, and response datasets (DNA_REPi) to new variables
exprs_z = GDSCE
mut = GDSCM
responses = GDSCR


# select drugs in GDSC response dataset (DNA_REPi)
drugs = set(responses["drug"].values)
print("Drugs:", drugs)


# convert GDSC expression, mutation, and response indices to string
GDSCE.index = GDSCE.index.astype(str)
GDSCM.index = GDSCM.index.astype(str)
responses.index = responses.index.astype(str)


# subset GDSC expression and mutation datasets (DNA_REPi) as to drugs
expression_zscores = []
mutations = []
for drug in drugs:
    samples = responses.loc[responses["drug"] == drug, :].index.values
    e_z = exprs_z.loc[samples, :]
    m = mut.loc[samples, :]
    expression_zscores.append(e_z)
    mutations.append(m)

GDSCEv2 = pd.concat(expression_zscores, axis=0)
GDSCMv2 = pd.concat(mutations, axis=0)
GDSCRv2 = responses


# filter DEGs from all genes in GDSC expression dataset (DNA_REPi) 
ls5 = list(set(GDSCE.columns).intersection(set(DEGs_entrez_id.astype(int))))

# filter shared samples between the subsetted GDSC expression and mutation datasets (DNA_REPi) 
ls6 = GDSCEv2.index.intersection(GDSCMv2.index)


# subset shared genes and samples in the subsetted GDSC expression, mutation, and response datasets (DNA_REPi) 
GDSCEv2 = GDSCEv2.loc[ls6, ls5]
GDSCMv2 = GDSCMv2.loc[ls6, ls5]
GDSCRv2 = GDSCRv2.loc[ls6, :]

PDXEgemcitabine = PDXEgemcitabine.loc[:,ls5]
PDXMgemcitabine = PDXMgemcitabine.loc[:,ls5]

TCGAEgemcitabine = TCGAEgemcitabine.loc[:,ls5]
TCGAMgemcitabine = TCGAMgemcitabine.loc[:,ls5]

responses.index = responses.index.values + "_" + responses["drug"].values


print(f"GDSC # of common samples and genes (Expression): {GDSCEv2.shape}")
print(f"GDSC # of common samples and genes (Mutation): {GDSCMv2.shape}")
print(f"GDSC # of common samples (Response): {GDSCRv2.shape[0]}\n")

print(f"PDX # of common samples and genes for Gemcitabine (Expression): {PDXEgemcitabine.shape}")
print(f"PDX # of common samples and genes for Gemcitabine (Mutation): {PDXMgemcitabine.shape}\n")

print(f"TCGA # of common samples and genes for Gemcitabine (Expression): {TCGAEgemcitabine.shape}")
print(f"TCGA # of common samples and genes for Gemcitabine (Mutation): {TCGAMgemcitabine.shape}\n")


# assign GDSC response dataset values (DNA_REPi) 
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
    lrm = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate1 = random.choice(ls_rate)
    rate2 = random.choice(ls_rate)
    rate3 = random.choice(ls_rate)
    wd = random.choice(ls_wd)

    print(
        f'\nmb_size = {mbs},  h_dim[1,2] = ({hdm1},{hdm2}), lr[E, M] = ({lre}, {lrm}), epoch = {epch}, rate[1,2,3] = ({rate1},{rate2},{rate3}), wd = {wd}, lrCL = {lrCL}\n')

    for train_index, test_index in skf.split(GDSCEv2.values, Y):
        k = k + 1

        X_trainE = GDSCEv2.values[train_index, :]
        X_testE = GDSCEv2.values[test_index, :]
        X_trainM = GDSCMv2.values[train_index, :]
        X_testM = GDSCMv2.values[test_index, :]
        y_trainE = Y[train_index]
        y_testE = Y[test_index]

        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)

        X_trainM = np.nan_to_num(X_trainM)
        X_testM = np.nan_to_num(X_testM)

        TX_testE = torch.FloatTensor(X_testE)
        TX_testM = torch.FloatTensor(X_testM)
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
                                                      torch.FloatTensor(X_trainM), torch.FloatTensor(y_trainE.astype(int)))

        trainLoader = torch.utils.data.DataLoader(
            dataset=trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler=sampler)

        n_sampE, IE_dim = X_trainE.shape
        n_sampM, IM_dim = X_trainM.shape

        h_dim1 = hdm1
        h_dim2 = hdm2
        Z_in = h_dim1 + h_dim2
        lrE = lre
        lrM = lrm
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

        class AEM(nn.Module):
            def __init__(self):
                super(AEM, self).__init__()
                self.EnM = torch.nn.Sequential(
                    nn.Linear(IM_dim, h_dim2),
                    nn.BatchNorm1d(h_dim2),
                    nn.ReLU(),
                    nn.Dropout(rate2))

            def forward(self, x):
                output = self.EnM(x)
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
        AutoencoderM = AEM()

        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
        solverM = optim.Adagrad(AutoencoderM.parameters(), lr=lrM)

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
                AutoencoderM.train()
                Clas.train()

                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    ZEX = AutoencoderE(dataE)
                    ZMX = AutoencoderM(dataC)

                    ZT = torch.cat((ZEX, ZMX), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    Pred = Clas(ZT)

                    loss = C_loss(Pred, target.view(-1, 1))

                    y_true = target.view(-1, 1)
                    y_pred = Pred
                    AUC = roc_auc_score(
                        y_true.detach().numpy(), y_pred.detach().numpy())

                    solverE.zero_grad()
                    solverM.zero_grad()
                    SolverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    solverM.step()
                    SolverClass.step()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                    epoch_cost3.append(AUC)
                    flag = 1

            if flag == 1:
                costtr.append(torch.mean(epoch_cost4))
                auctr.append(np.mean(epoch_cost3))

            with torch.no_grad():

                AutoencoderE.eval()
                AutoencoderM.eval()
                Clas.eval()

                ZET = AutoencoderE(TX_testE)
                ZMT = AutoencoderM(TX_testM)

                ZTT = torch.cat((ZET, ZMT), 1)
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

        plt.plot(np.squeeze(costtr), '-r',
                 np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Gemcitabine iter = {}, fold = {}, mb_size = {},  h_dim[1,2] = ({},{}, \nlr[E,M] = ({}, {}), epoch = {}, rate[1,2,3] = ({},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, lre, lrm,
                   epch, rate1, rate2, rate3, wd, lrCL)

        plt.suptitle(title,  fontsize=8, fontweight='bold')
        plt.savefig(save_results_to + title + '.png', dpi=150)
        plt.close()

        plt.plot(np.squeeze(auctr), '-r', np.squeeze(aucts), '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        title = 'AUC Gemcitabine iter = {}, fold = {}, mb_size = {},  h_dim[1,2] = ({},{}, \nlr[E,M] = ({}, {}), epoch = {}, rate[1,2,3] = ({},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, lre, lrm,
                   epch, rate1, rate2, rate3, wd, lrCL)

        plt.suptitle(title,  fontsize=8, fontweight='bold')
        plt.savefig(save_results_to + title + '.png', dpi=150)
        plt.close()


# save models
os.makedirs(save_models_to, exist_ok=True)
os.chdir(save_models_to)
torch.save(
    AutoencoderE, "Exprs_DNA_REPi_Expression_Mutation_GDSC_PDX_TCGA_Gemcitabine_Third_Strategy.pt")
torch.save(AutoencoderM, "Mut_DNA_REPi_Expression_Mutation_GDSC_PDX_TCGA_Gemcitabine_Third_Strategy.pt")
torch.save(Clas, 'Class_DNA_REPi_Expression_Mutation_GDSC_PDX_TCGA_Gemcitabine_Third_Strategy.pt')


# end time
end = time.time()
print(f"\nElapsed time: {(end - start)/3600} hours")
