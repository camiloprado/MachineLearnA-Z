import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import kagglehub, os

#Baixa o arquivo de dados de credito do Kaggle
var_strPath = kagglehub.dataset_download(handle="laotse/credit-risk-dataset")

#Define a opção de não truncar as tabelas
pd.set_option('display.max_columns', None)

#Desabilita a notação científica
# pd.set_option('display.float_format', '{:.2f}'.format)

#Define o nome do arquivo
var_strFileName = "credit_risk_dataset.csv"

#Cria o caminho completo do arquivo
var_strPathCredit = os.path.join(var_strPath, var_strFileName)

#Recebe o arquivo e armazena em um DataFrame
var_tblCredit = pd.read_csv(var_strPathCredit)
# print(f"Tabela inteira: \n {var_tblCredit}")

#Exibe as primeiras 10 linhas do DataFrame
var_tblPrimeiros = var_tblCredit.head(10)
# print(f"Dez primeiros valores: \n{var_tblPrimeiros}")

#Exibe as últimas 8 linhas do DataFrame
var_tblUltimos = var_tblCredit.tail(8)
# print(f"Oito ultimos valores: \n{var_tblUltimos}")

#Exibe o número de linhas e colunas do DataFrame
var_tblDescricao = var_tblCredit.describe()
# print(f"Valores especificos: \n{var_tblDescricao}")

#Coleta toda a informação da coluna 'person_income'
var_colIncome = var_tblCredit['person_income']

#Coleta a informação onde o valor é maior ou igual a 6000000.00
var_intMaiorLoan = var_tblCredit[var_colIncome >= 6000000.00]
# print(f"Valores maiores que 6000000.00: \n{var_intMaiorLoan}")

#Coleta toda a informação da coluna 'loan_amnt'
var_colIncome = var_tblCredit['loan_amnt']

#Coleta a informação onde o valor é menor ou igual a 4000.00
var_intMenorLoan = var_tblCredit[var_colIncome <= 500.00]
# print(f"Valores menores que 4000.00: \n{var_intMenorLoan}")

#Conta os valores unicos da coluna 'cb_person_default_on_file'
var_intValoresUnicos = np.unique(var_tblCredit['cb_person_default_on_file'], return_counts=True)

#Cria um gráfico de barras com a contagem dos valores unicos
sns.countplot(x='cb_person_default_on_file', data=var_tblCredit)
# plt.show()

#Cria um gráfico de barras com a contagem dos valores de idade
plt.hist(x=var_tblCredit['person_age'])
# plt.show()

#Criar um gráfico de barras com a contagem dos valores de empréstimo
plt.hist(x=var_tblCredit['person_income'])
# plt.show()

#Criar um gráfico de barras com a contagem dos valores de dividas
plt.hist(x=var_tblCredit['loan_amnt'])
# plt.show()

var_grfGrafico = px.scatter_matrix(var_tblCredit, dimensions=['person_age', 'person_income', 'loan_amnt'], color='cb_person_default_on_file')
var_grfGrafico.show()