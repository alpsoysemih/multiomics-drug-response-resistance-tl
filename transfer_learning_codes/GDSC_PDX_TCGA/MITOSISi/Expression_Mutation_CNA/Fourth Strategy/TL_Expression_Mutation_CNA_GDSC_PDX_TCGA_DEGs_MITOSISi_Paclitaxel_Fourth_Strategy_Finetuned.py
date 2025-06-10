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
import scipy.stats as stats
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
import warnings

# suppress all warnings
warnings.filterwarnings("ignore")


# start time
start = time.time()


# define directories
cell_line_dir = "/arf/home/salpsoy/Thesis_Work/Supplementary_Files/GDSC/"
dataset_dir = "/arf/home/salpsoy/Thesis_Work/Datasets/"
DEGs_dir = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/MITOSISi/"
load_models_from = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/MITOSISi/save_models/GDSC_PDX_TCGA_Fourth_Strategy/Expression_Mutation_CNA/"
save_finetuned_models_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/MITOSISi/save_models/GDSC_PDX_TCGA_Fourth_Strategy/Expression_Mutation_CNA/Fourth_Strategy_Finetuned_Models/"
save_figures_to = "/arf/home/salpsoy/Thesis_Work/Results/GDSC_DEGs_inhibitors/MITOSISi/save_models/GDSC_PDX_TCGA_Fourth_Strategy/Expression_Mutation_CNA/AUC and Cost Plots (Finetuned)/"


# define maximum iteration and set random seed
max_iter = 50
torch.manual_seed(42)
random.seed(42)


# change directory to read cell line details in GDSC
os.chdir(cell_line_dir)


# read cell line details in GDSC 
GDSC_cell_line_details = pd.read_excel(
    "GDSC_Cell_Lines_Details.xlsx", keep_default_na=False)
GDSC_cell_line_details.set_index("COSMIC identifier", inplace=True)
GDSC_cell_line_details = GDSC_cell_line_details.iloc[:-1, ]
GDSC_cell_line_details.index = GDSC_cell_line_details.index.astype(str)


# change directory to read DEGs
os.chdir(DEGs_dir)


# read diferentially expressed genes common in mitotic inhibitors
# exclude paclitaxel-unique DEGs from unionized features
DEGs_filtered_data = pd.read_excel("MITOSISi_Differentially_Expressed_Genes (EnsemblID).xlsx",
                                   sheet_name="Common DEGs")
filter = DEGs_filtered_data["Frequency"] == 1
DEGs_freq_one =  DEGs_filtered_data[filter]
DEGs_freq_one.head()
DEGs_Paclitaxel_data = pd.read_excel("MITOSISi_Differentially_Expressed_Genes (EnsemblID).xlsx",
                                              sheet_name = "Paclitaxel")
filter = DEGs_Paclitaxel_data["Gene.Symbol"].isin(DEGs_freq_one["Gene Symbol"])
only_paclitaxel_degs = DEGs_Paclitaxel_data.loc[filter, "Gene.Symbol"]
filter = DEGs_filtered_data["Gene Symbol"].isin(only_paclitaxel_degs)
DEGs_filtered_data = DEGs_filtered_data[~filter]


# get Entrez IDs from gene symbols
mg = mygene.MyGeneInfo()
DEGs_entrez_id = mg.querymany(DEGs_filtered_data["Gene Symbol"],
                              species="human",
                              scopes="symbol",
                              field="entrezgene",
                              as_dataframe=True)["entrezgene"]
DEGs_entrez_id.dropna(inplace=True)
DEGs_entrez_id = pd.Series(DEGs_entrez_id)
result = mg.query("SLC22A18", species="human", scopes="symbol", fields="entrezgene")
DEGs_entrez_id = pd.concat([DEGs_entrez_id, pd.Series(str(result["hits"][1]["entrezgene"]))], ignore_index=True)


# change directory to read multi-omics datasets
os.chdir(dataset_dir)


# read GDSC expression dataset (MITOSISi)
# remove samples with shared cancer types from training dataset
GDSCE = pd.read_csv("GDSC_exprs.z.MITOSISi.tsv",
                    sep="\t",
                    index_col=0,
                    decimal=",")
GDSCE = pd.DataFrame.transpose(GDSCE)
GDSCE = pd.merge(GDSC_cell_line_details,
                 GDSCE,
                 left_index=True,
                 right_index=True)
filter = (GDSCE["GDSC\nTissue descriptor 1"] != "breast") & \
         (GDSCE["GDSC\nTissue descriptor 1"] != "lung_NSCLC") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "HNSC") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "STAD") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "UCS") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "LUAD") & \
         (GDSCE["Cancer Type\n(matching TCGA label)"] != "BRCA")
GDSCE = GDSCE.loc[filter, ]
GDSCE.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCE.index = GDSCE.index.astype(int)


# read GDSC mutation dataset (MITOSISi)
# remove samples with shared cancer types from training dataset
GDSCM = pd.read_csv("GDSC_mutations.MITOSISi.tsv",
                    sep="\t", index_col=0, decimal=".")
GDSCM.drop_duplicates(keep='last')
GDSCM = pd.DataFrame.transpose(GDSCM)
GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]
GDSCM = pd.merge(GDSC_cell_line_details,
                 GDSCM,
                 left_index=True,
                 right_index=True)
filter = (GDSCM["GDSC\nTissue descriptor 1"] != "breast") & \
         (GDSCM["GDSC\nTissue descriptor 1"] != "lung_NSCLC") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "HNSC") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "STAD") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "UCS") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "LUAD") & \
         (GDSCM["Cancer Type\n(matching TCGA label)"] != "BRCA")
GDSCM = GDSCM.loc[filter, ]
GDSCM.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCM.index = GDSCM.index.astype(int)


# read GDSC CNA dataset (MITOSISi)
# remove samples with shared cancer types from training dataset
GDSCC = pd.read_csv("GDSC_CNA.MITOSISi.tsv",
                    sep="\t", index_col=0, decimal=".")
GDSCC.drop_duplicates(keep='last')
GDSCC = pd.DataFrame.transpose(GDSCC)
GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]
GDSCC = pd.merge(GDSC_cell_line_details,
                 GDSCC,
                 left_index=True,
                 right_index=True)
filter = (GDSCC["GDSC\nTissue descriptor 1"] != "breast") & \
         (GDSCC["GDSC\nTissue descriptor 1"] != "lung_NSCLC") & \
         (GDSCC["Cancer Type\n(matching TCGA label)"] != "HNSC") & \
         (GDSCC["Cancer Type\n(matching TCGA label)"] != "STAD") & \
         (GDSCC["Cancer Type\n(matching TCGA label)"] != "UCS") & \
         (GDSCC["Cancer Type\n(matching TCGA label)"] != "LUAD") & \
         (GDSCC["Cancer Type\n(matching TCGA label)"] != "BRCA")
GDSCC = GDSCC.loc[filter, ]
GDSCC.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCC.index = GDSCC.index.astype(int)


# read GDSC response dataset (MITOSISi) and binarize 
# remove samples with shared cancer types from training dataset
GDSCR = pd.read_csv("GDSC_response.MITOSISi.tsv",
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
filter = (GDSCR["GDSC\nTissue descriptor 1"] != "breast") & \
         (GDSCR["GDSC\nTissue descriptor 1"] != "lung_NSCLC") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "HNSC") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "STAD") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "UCS") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "LUAD") & \
         (GDSCR["Cancer Type\n(matching TCGA label)"] != "BRCA")
GDSCR = GDSCR.loc[filter, ]
GDSCR.drop(GDSC_cell_line_details.columns, axis=1, inplace=True)
GDSCR = GDSCR.loc[GDSCR["drug"] != "Paclitaxel", ]
GDSCR.index = GDSCR.index.astype(int)


# Read PDX expression dataset (Paclitaxel)
PDXEpaclitaxel = pd.read_csv("PDX_exprs.Paclitaxel.eb_with.GDSC_exprs.Paclitaxel.tsv",
                      sep="\t", index_col=0, decimal=",")
PDXEpaclitaxel = pd.DataFrame.transpose(PDXEpaclitaxel)
PDXEpaclitaxel.head(3)

# Read TCGA expression dataset (Paclitaxel)
TCGAEpaclitaxel = pd.read_csv("TCGA_exprs.Paclitaxel.eb_with.GDSC_exprs.Paclitaxel.tsv",
                       sep="\t", index_col=0, decimal=",")
TCGAEpaclitaxel = pd.DataFrame.transpose(TCGAEpaclitaxel)
TCGAEpaclitaxel.head(3)


# Read PDX mutation dataset (Paclitaxel)
PDXMpaclitaxel = pd.read_csv("PDX_mutations.Paclitaxel.tsv",
                      sep="\t", index_col=0, decimal=",")
PDXMpaclitaxel.drop_duplicates(keep='last')
PDXMpaclitaxel = pd.DataFrame.transpose(PDXMpaclitaxel)
PDXMpaclitaxel = PDXMpaclitaxel.loc[:, ~PDXMpaclitaxel.columns.duplicated()]
PDXMpaclitaxel.columns = PDXMpaclitaxel.columns.astype(int)


# Read TCGA mutation dataset (Paclitaxel)
TCGAMpaclitaxel = pd.read_csv("TCGA_mutations.Paclitaxel.tsv",
                       sep="\t", index_col=0, decimal=",")
TCGAMpaclitaxel.drop_duplicates(keep='last')
TCGAMpaclitaxel = pd.DataFrame.transpose(TCGAMpaclitaxel)
TCGAMpaclitaxel = TCGAMpaclitaxel.loc[:, ~TCGAMpaclitaxel.columns.duplicated()]


# Read PDX CNA dataset (Paclitaxel)
PDXCpaclitaxel = pd.read_csv("PDX_CNA.Paclitaxel.tsv",
                      sep="\t", index_col=0, decimal=",")
PDXCpaclitaxel.drop_duplicates(keep='last')
PDXCpaclitaxel = pd.DataFrame.transpose(PDXCpaclitaxel)
PDXCpaclitaxel = PDXCpaclitaxel.loc[:, ~PDXCpaclitaxel.columns.duplicated()]

# Read TCGA CNA dataset (Paclitaxel)
TCGACpaclitaxel = pd.read_csv("TCGA_CNA.Paclitaxel.tsv",
                       sep="\t", index_col=0, decimal=",")
TCGACpaclitaxel.drop_duplicates(keep='last')
TCGACpaclitaxel = pd.DataFrame.transpose(TCGACpaclitaxel)
TCGACpaclitaxel = TCGACpaclitaxel.loc[:, ~TCGACpaclitaxel.columns.duplicated()]

# variance threshold for GDSC expression dataset (MITOSISi)
selector = VarianceThreshold(0.05)
selector.fit_transform(GDSCE)
GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]


# fill NA values and binarize GDSC mutation and CNA datasets (MITOSISi)
GDSCM = GDSCM.fillna(0)
GDSCC = GDSCC.fillna(0)
GDSCM[GDSCM != 0.0] = 1
GDSCC[GDSCC != 0.0] = 1


# select shared genes between GDSC, PDX, and TCGA datasets
ls = GDSCE.columns.intersection(GDSCM.columns)
ls = ls.intersection(GDSCC.columns)
ls = ls.intersection(PDXEpaclitaxel.columns)
ls = ls.intersection(PDXMpaclitaxel.columns)
ls = ls.intersection(PDXCpaclitaxel.columns)
ls = ls.intersection(TCGAEpaclitaxel.columns)
ls = ls.intersection(TCGAMpaclitaxel.columns)
ls = ls.intersection(TCGACpaclitaxel.columns)
ls = pd.unique(ls)



# select shared samples between GDSC expression, mutation, CNA, and response datasets (MITOSISi)
ls2 = GDSCE.index.intersection(GDSCM.index)
ls2 = ls2.intersection(GDSCC.index)
ls2 = ls2.intersection(GDSCR.index)


# subset shared genes and samples in GDSC expression, mutation, CNA, and response datasets (MITOSISi)
GDSCE = GDSCE.loc[ls2, ls]
GDSCM = GDSCM.loc[ls2, ls]
GDSCC = GDSCC.loc[ls2, ls]
GDSCR = GDSCR.loc[ls2, :]


# select shared samples between PDX expression, mutation, and CNA datasets (Paclitaxel)
ls3 = PDXEpaclitaxel.index.intersection(PDXMpaclitaxel.index)
ls3 = ls3.intersection(PDXCpaclitaxel.index)


# subset shared samples and genes in PDX expression, mutation, and CNA datasets (Paclitaxel)
PDXEpaclitaxel = PDXEpaclitaxel.loc[ls3, ls]
PDXMpaclitaxel = PDXMpaclitaxel.loc[ls3, ls]
PDXCpaclitaxel = PDXCpaclitaxel.loc[ls3, ls]


# select shared samples between TCGA expression, mutation, and CNA datasets (Paclitaxel)
ls4 = TCGAEpaclitaxel.index.intersection(TCGAMpaclitaxel.index)
ls4 = ls4.intersection(TCGAMpaclitaxel.index)



# subset shared samples and genes in TCGA expression, mutation, and CNA datasets (Paclitaxel)
TCGAEpaclitaxel = TCGAEpaclitaxel.loc[ls4, ls]
TCGAMpaclitaxel = TCGAMpaclitaxel.loc[ls4, ls]
TCGACpaclitaxel = TCGACpaclitaxel.loc[ls4, ls]


# assign GDSC expression, mutation, and response datasets (MITOSISi) to new variables
exprs_z = GDSCE
mut = GDSCM
cna = GDSCC
responses = GDSCR


# select drugs in GDSC response dataset (MITOSISi)
drugs = set(responses["drug"].values)
print("Drugs:", drugs)


# convert the indices of GDSC datasets to string
GDSCE.index = GDSCE.index.astype(str)
GDSCM.index = GDSCM.index.astype(str)
GDSCC.index = GDSCC.index.astype(str)
responses.index = responses.index.astype(str)


# subset GDSC expression, mutation, and CNA datasets (MITOSISi) as to drugs
expression_zscores = []
mutations = []
CNA = []
for drug in drugs:
    samples = responses.loc[responses["drug"] == drug, :].index.values
    e_z = exprs_z.loc[samples, :]
    m = mut.loc[samples, :]
    c = cna.loc[samples, :]
    expression_zscores.append(e_z)
    mutations.append(m)
    CNA.append(c)

GDSCEv2 = pd.concat(expression_zscores, axis=0)
GDSCMv2 = pd.concat(mutations, axis=0)
GDSCCv2 = pd.concat(CNA, axis=0)
GDSCRv2 = responses



# filter DEGs from all genes in GDSC expression dataset (MITOSISi) 
ls5 = list(set(GDSCE.columns).intersection(set(DEGs_entrez_id.astype(int))))

# filter shared samples between the subsetted GDSC expression, mutation, and CNA datasets (MITOSISi) 
ls6 = GDSCEv2.index.intersection(GDSCMv2.index)
ls6 = ls6.intersection(GDSCCv2.index)


# subset shared genes and samples in the subsetted GDSC expression, mutation, CNA and response datasets (MITOSISi) 
GDSCEv2 = GDSCEv2.loc[ls6, ls5]
GDSCMv2 = GDSCMv2.loc[ls6, ls5]
GDSCCv2 = GDSCCv2.loc[ls6, ls5]
GDSCRv2 = GDSCRv2.loc[ls6, :]

PDXEpaclitaxel = PDXEpaclitaxel.loc[:,ls5]
PDXMpaclitaxel = PDXMpaclitaxel.loc[:,ls5]
PDXCpaclitaxel = PDXCpaclitaxel.loc[:,ls5]

TCGAEpaclitaxel = TCGAEpaclitaxel.loc[:,ls5]
TCGAMpaclitaxel = TCGAMpaclitaxel.loc[:,ls5]
TCGACpaclitaxel = TCGACpaclitaxel.loc[:,ls5]

responses.index = responses.index.values + "_" + responses["drug"].values

print(f"GDSC # of common samples and genes (Expression): {GDSCEv2.shape}")
print(f"GDSC # of common samples and genes (Mutation): {GDSCMv2.shape}")
print(f"GDSC # of common samples and genes (CNA): {GDSCCv2.shape}")
print(f"GDSC # of common samples (Response): {GDSCRv2.shape[0]}\n")

print(f"PDX # of common samples and genes for Paclitaxel (Expression): {PDXEpaclitaxel.shape}")
print(f"PDX # of common samples and genes for Paclitaxel (Mutation): {PDXMpaclitaxel.shape}\n")
print(f"PDX # of common samples and genes for Paclitaxel (CNA): {PDXCpaclitaxel.shape}\n")

print(f"TCGA # of common samples and genes for Paclitaxel (Expression): {TCGAEpaclitaxel.shape}")
print(f"TCGA # of common samples and genes for Paclitaxel (Mutation): {TCGAMpaclitaxel.shape}\n")
print(f"TCGA # of common samples and genes for Paclitaxel (CNA): {TCGACpaclitaxel.shape}\n")

# assign GDSC response dataset values (MITOSISi) 
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


# freeze expression and mutation layers, fine-tune only classification layer
for iters in range(max_iter):
    k = 0
    mbs = random.choice(ls_mb_size)
    hdm1 = random.choice(ls_h_dim)
    hdm2 = random.choice(ls_h_dim)
    hdm3 = random.choice(ls_h_dim)
    lre = random.choice(ls_lr)
    lrm = random.choice(ls_lr)
    lrc = random.choice(ls_lr)
    lrCL = random.choice(ls_lr)
    epch = random.choice(ls_epoch)
    rate1 = random.choice(ls_rate)
    rate2 = random.choice(ls_rate)
    rate3 = random.choice(ls_rate)
    rate4 = random.choice(ls_rate)
    wd = random.choice(ls_wd)

    print(
        f'\nmb_size = {mbs},  h_dim[1,2,3] = ({hdm1},{hdm2},{hdm3}), lr[E, M, C] = ({lre}, {lrm}, {lrc}), epoch = {epch}, rate[1,2,3,4] = ({rate1},{rate2},{rate3},{rate4}), wd = {wd}, lrCL = {lrCL}\n')

    for train_index, test_index in skf.split(GDSCEv2.values, Y):
        k = k + 1

        X_trainE = GDSCEv2.values[train_index, :]
        X_testE = GDSCEv2.values[test_index, :]
        X_trainM = GDSCMv2.values[train_index, :]
        X_testM = GDSCMv2.values[test_index, :]
        X_trainC = GDSCCv2.values[train_index, :]
        X_testC = GDSCCv2.values[test_index, :]
        y_trainE = Y[train_index]
        y_testE = Y[test_index]

        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)

        X_trainM = np.nan_to_num(X_trainM)
        X_testM = np.nan_to_num(X_testM)

        X_trainC = np.nan_to_num(X_trainC)
        X_testC = np.nan_to_num(X_testC)

        TX_testE = torch.FloatTensor(X_testE)
        TX_testM = torch.FloatTensor(X_testM)
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
                                                      torch.FloatTensor(X_trainM), 
                                                      torch.FloatTensor(X_trainC), torch.FloatTensor(y_trainE.astype(int)))

        trainLoader = torch.utils.data.DataLoader(
            dataset=trainDataset, batch_size=mb_size, shuffle=False, num_workers=1, sampler=sampler)

        n_sampE, IE_dim = X_trainE.shape
        n_sampM, IM_dim = X_trainM.shape
        n_sampC, IC_dim = X_trainC.shape

        h_dim1 = hdm1
        h_dim2 = hdm2
        h_dim3 = hdm3
        Z_in = h_dim1 + h_dim2 + h_dim3
        lrE = lre
        lrm = lrm
        lrc = lrc
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

        class AEC(nn.Module):
            def __init__(self):
                super(AEC, self).__init__()
                self.EnC = torch.nn.Sequential(
                    nn.Linear(IC_dim, h_dim3),
                    nn.BatchNorm1d(h_dim3),
                    nn.ReLU(),
                    nn.Dropout(rate3))

            def forward(self, x):
                output = self.EnC(x)
                return output

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.FC = torch.nn.Sequential(
                    nn.Linear(Z_in, 1),
                    nn.Dropout(rate4),
                    nn.Sigmoid())

            def forward(self, x):
                return self.FC(x)

        torch.cuda.manual_seed_all(42)

        AutoencoderE = AEE()
        AutoencoderM = AEM()
        AutoencoderC = AEC()
        Clas = Classifier()

        AutoencoderE = torch.load(
            load_models_from + 'Exprs_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy.pt')
        AutoencoderM = torch.load(
            load_models_from + 'Mut_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy.pt')
        AutoencoderC = torch.load(
            load_models_from + 'CNA_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy.pt')
        Clas = torch.load(
            load_models_from + 'Class_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy.pt')

        solverE = optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
        solverM = optim.Adagrad(AutoencoderM.parameters(), lr=lrm)
        solverC = optim.Adagrad(AutoencoderC.parameters(), lr=lrc)

        solverClass = optim.Adagrad(
            Clas.parameters(), lr=lrCL, weight_decay=wd)
        C_loss = torch.nn.BCELoss()

        for it in range(epoch):
            epoch_cost4 = 0
            epoch_cost3 = []
            num_minibatches = int(n_sampE / mb_size)

            for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                flag = 0
                AutoencoderE.train()
                AutoencoderM.train()
                AutoencoderC.train()
                Clas.train()

                for param in AutoencoderE.parameters():
                    param.requires_grad = False
                for param in AutoencoderM.parameters():
                    param.requires_grad = False
                for param in AutoencoderC.parameters():
                    param.requires_grad = False
                for param in Clas.parameters():
                    param.requires_grad = True


                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    ZEX = AutoencoderE(dataE)
                    ZMX = AutoencoderM(dataM)
                    ZCX = AutoencoderC(dataC)

                    ZT = torch.cat((ZEX, ZMX, ZCX), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    Pred = Clas(ZT)

                    loss = C_loss(Pred, target.view(-1, 1))

                    y_true = target.view(-1, 1)
                    y_pred = Pred
                    AUC = roc_auc_score(
                        y_true.detach().numpy(), y_pred.detach().numpy())

                    solverE.zero_grad()
                    solverM.zero_grad()
                    solverC.zero_grad()
                    solverClass.zero_grad()

                    loss.backward()

                    solverE.step()
                    solverM.step()
                    solverC.step()
                    solverClass.step()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                    epoch_cost3.append(AUC)
                    flag = 1

            if flag == 1:
                costtr.append(torch.mean(epoch_cost4))
                auctr.append(np.mean(epoch_cost3))

            with torch.no_grad():

                AutoencoderE.eval()
                AutoencoderM.eval()
                AutoencoderC.eval()
                Clas.eval()

                ZET = AutoencoderE(TX_testE)
                ZMT = AutoencoderM(TX_testM)
                ZCT = AutoencoderC(TX_testC)

                ZTT = torch.cat((ZET, ZMT, ZCT), 1)
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

        os.makedirs(save_figures_to, exist_ok=True)

        plt.plot(np.squeeze(costtr), '-r', np.squeeze(costts), '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        title = 'Cost Paclitaxel iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), \nlr[E,M,C] = ({}, {},{}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, hdm3, lre, lrm, lrc,
                   epch, rate1, rate2, rate3, rate4, wd, lrCL)

        plt.suptitle(title, fontsize=8,  fontweight='bold')
        plt.savefig(save_figures_to + title + '.png', dpi=150)
        plt.close()

        plt.plot(np.squeeze(auctr), '-r', np.squeeze(aucts), '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        title = 'AUC Paclitaxel iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), \nlr[E,M,C] = ({}, {},{}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}'.\
            format(iters, k, mbs, hdm1, hdm2, hdm3, lre, lrm, lrc,
                   epch, rate1, rate2, rate3, rate4, wd, lrCL)

        plt.suptitle(title, fontsize=8,  fontweight='bold')
        plt.savefig(save_figures_to + title + '.png', dpi=150)
        plt.close()

    os.makedirs(save_finetuned_models_to, exist_ok=True)
    os.chdir(save_finetuned_models_to)
    torch.save(AutoencoderE,
               f"Exprs_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy_Finetuned_{iters + 1}.pt")
    torch.save(
        AutoencoderM, f"Mut_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy_Finetuned_{iters + 1}.pt")
    torch.save(
        AutoencoderC, f"CNA_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy_Finetuned_{iters + 1}.pt")    
    torch.save(Clas, f'Class_MITOSISi_Expression_Mutation_CNA_GDSC_PDX_TCGA_Paclitaxel_Fourth_Strategy_Finetuned_{iters + 1}.pt')
