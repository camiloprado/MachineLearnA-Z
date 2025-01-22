from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import kagglehub, os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, plotly.express as px, pickle

#Credito
var_objScalerCredit = StandardScaler()

#Recebe o arquivo e armazena em um DataFrame
var_tblCredit = pd.read_csv("Machine Learning e Data Science com Python de A à Z\Bases de dados\credit_data.csv")

var_tblMenorZero = var_tblCredit.loc[var_tblCredit['age'] < 0]
# print(f"Valores menores que 0: \n{var_tblMenorZero}")

# Apagando a coluna inteira
var_tblCredit2 = var_tblCredit.drop(labels='age', axis=1)
# print(var_tblCredit2)

# Apagando somente os registros com valores inconsistentes
var_tblCredit3 = var_tblCredit.drop(var_tblCredit[var_tblCredit['age'] < 0].index)
# print(var_tblCredit3)

# Preencher os valores inconsistente com a média das idades
var_tblCredit['age'][var_tblCredit['age'] > 0].mean()
var_tblTreino = var_tblCredit
var_tblTreino['age'].fillna(var_tblTreino['age'].mean(), inplace=True)
var_tblXCredit = var_tblTreino.iloc[:, 1:4].values
var_tblYCredit = var_tblTreino.iloc[:, :4].values
var_tblXCredit = var_objScalerCredit.fit_transform(var_tblXCredit)

#Census
var_CSVCensus = pd.read_csv(r'Machine Learning e Data Science com Python de A à Z\Bases de dados\census.csv')
# print(var_CSVCensus)
# print(var_CSVCensus.describe())
# print(var_CSVCensus.isnull().sum())
# print(np.unique(var_CSVCensus['income'], return_counts=True))

# sns.countplot(x=var_CSVCensus['income'])
# plt.hist(x=var_CSVCensus['age'])
# plt.hist(x=var_CSVCensus['education-num'])
# plt.hist(x=var_CSVCensus['hour-per-week'])
# plt.show()

# var_grfGrafico = px.treemap(var_CSVCensus, path=['workclass', 'age'])
# var_grfGrafico = px.treemap(var_CSVCensus, path=['occupation', 'relationship', 'age'])
# var_grfGrafico = px.parallel_categories(var_CSVCensus, dimensions=['occupation', 'relationship'])
# var_grfGrafico = px.parallel_categories(var_CSVCensus, dimensions=['workclass', 'occupation', 'income'])
# var_grfGrafico = px.parallel_categories(var_CSVCensus, dimensions=['education', 'income'])
# var_grfGrafico.show()

# print(var_CSVCensus.columns)

var_XCensus = var_CSVCensus.iloc[:, 0:14].values
# print(var_XCensus)

var_YCensus = var_CSVCensus.iloc[:, 14].values
# print(var_YCensus)

#Label Encoder -> String em Numeros
label_encoder_teste = LabelEncoder()

teste = label_encoder_teste.fit_transform(var_XCensus[:, 1])
# print(teste)

var_labWorkclass = LabelEncoder()
var_labEducation = LabelEncoder()
var_labMarital = LabelEncoder()
var_labOccupation = LabelEncoder()
var_labRelationship = LabelEncoder()
var_labRace = LabelEncoder()
var_labSex = LabelEncoder()
var_labCountry = LabelEncoder()

var_XCensus[:, 1] = var_labWorkclass.fit_transform(var_XCensus[:, 1])
var_XCensus[:, 3] = var_labEducation.fit_transform(var_XCensus[:, 3])
var_XCensus[:, 5] = var_labMarital.fit_transform(var_XCensus[:, 5])
var_XCensus[:, 6] = var_labOccupation.fit_transform(var_XCensus[:, 6])
var_XCensus[:, 7] = var_labRelationship.fit_transform(var_XCensus[:, 7])
var_XCensus[:, 8] = var_labRace.fit_transform(var_XCensus[:, 8])
var_XCensus[:, 9] = var_labSex.fit_transform(var_XCensus[:, 9])
var_XCensus[:, 13] = var_labCountry.fit_transform(var_XCensus[:, 13])

# var_XCensus.shape

#One Hot Encoder -> Variaveis Dummy
var_objOneHotEncoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
var_XCensus = var_objOneHotEncoder.fit_transform(var_XCensus).toarray()
# print(var_XCensus)
# var_XCensus.shape

#Escalonamento dos Valores -> Deixar na mesma escala
var_objScalerCensus = StandardScaler()
var_XCensus = var_objScalerCensus.fit_transform(var_XCensus)

#Divisão das bases de treinamento e teste
var_XTreinamentoCredit, var_XTesteCredit, var_YTreinamentoCredit, var_YTesteCredit = train_test_split(var_tblXCredit, var_tblYCredit, test_size=0.25, random_state=0)
print(var_XTreinamentoCredit.shape, var_YTreinamentoCredit.shape)
# int(var_XTesteCredit.shape, var_YTesteCredit.shape)
var_XTreinamentoCensus, var_XTesteCensus, var_YTreinamentoCensus, var_YTesteCensus = train_test_split(var_XCensus, var_YCensus, test_size=0.15, random_state=0)
print(var_XTreinamentoCensus.shape, var_YTreinamentoCensus.shape)
# int(var_XTesteCensus.shape, var_YTesteCensus.shape)

#Salvar as variáveis
with open('creditEstudo.pkl', mode='wb') as f:
    pickle.dump((var_XTreinamentoCredit, var_YTreinamentoCredit, var_XTesteCredit, var_YTesteCredit), f)


with open('censusEstudo.pkl', mode='wb') as f:
    pickle.dump((var_XTreinamentoCensus, var_YTreinamentoCensus, var_XTesteCensus, var_YTesteCensus), f)