from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import kagglehub, os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, plotly.express as px, pickle

#Risco de Crédito
var_naiveRiscoCredito = GaussianNB()
with open(r'Bases de dados\risco_credito.pkl', mode='rb') as f:
    var_XRiscoCredito, var_YRiscoCredito = pickle.load(f)
     
var_naiveRiscoCredito.fit(var_XRiscoCredito, var_YRiscoCredito)
var_listPrevisaoRiscoCredito = var_naiveRiscoCredito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

# print(var_listPrevisaoRiscoCredito)

# print(var_naiveRiscoCredito.classes_)
# print(var_naiveRiscoCredito.class_count_)
# print(var_naiveRiscoCredito.class_prior_)

# Base de Creditos
var_naiveCredito = GaussianNB()
with open(r'Bases de dados\credit.pkl', mode='rb') as f:
    var_XTreinamentoCredit, var_YTreinamentoCredit, var_XTesteCredit, var_YTesteCredit = pickle.load(f)

var_naiveCredito.fit(var_XTreinamentoCredit, var_YTreinamentoCredit)
var_listPrevisaoCredito = var_naiveCredito.predict(var_XTesteCredit)
# print(var_listPrevisaoCredito)
# print(var_YTreinamentoCredit)
var_intScore = accuracy_score(var_YTesteCredit, var_listPrevisaoCredito)
# print(var_intScore)

var_listConfusion = confusion_matrix(var_YTesteCredit, var_listPrevisaoCredito)
cm = ConfusionMatrix(var_naiveCredito)
cm.fit(var_XTreinamentoCredit, var_YTreinamentoCredit)
cm.score(var_XTesteCredit, var_YTesteCredit)
cm.show()

print("Relatório de Classificação dos Creditos: ")
print(classification_report(var_YTesteCredit, var_listPrevisaoCredito))

# Base Census
var_naiveCensus = GaussianNB()
with open(r'Bases de dados\census.pkl', mode='rb') as f:
    var_XTreinamentoCensus, var_YTreinamentoCensus, var_XTesteCensus, var_YTesteCensus = pickle.load(f)

var_naiveCensus.fit(var_XTreinamentoCensus, var_YTreinamentoCensus)
var_listPrevisaoCensus = var_naiveCensus.predict(var_XTesteCensus)
var_intScore = accuracy_score(var_YTesteCensus, var_listPrevisaoCensus)
# print(var_intScore)

var_listConfusion = confusion_matrix(var_YTesteCensus, var_listPrevisaoCensus)
cm = ConfusionMatrix(var_naiveCensus)
cm.fit(var_XTreinamentoCensus, var_YTreinamentoCensus)
cm.score(var_XTesteCensus, var_YTesteCensus)
cm.show()

print("Relatório de Classificação dos Census: ")
print(classification_report(var_YTesteCensus, var_listPrevisaoCensus))
