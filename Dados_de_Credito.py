from sklearn.preprocessing import StandardScaler
import kagglehub, os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, plotly.express as px


var_objScalerCredit = StandardScaler()

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
# var_grfGrafico.show()

var_tblMenorZero = var_tblCredit.loc[var_tblCredit['person_age'] < 0]
# print(f"Valores menores que 0: \n{var_tblMenorZero}")

# Apagando a coluna inteira
var_tblCredit2 = var_tblCredit.drop(labels='person_age', axis=1)
# print(var_tblCredit2)

# Apagando somente os registros com valores inconsistentes
var_tblCredit3 = var_tblCredit.drop(var_tblCredit[var_tblCredit['person_age'] < 0].index)
# print(var_tblCredit3)

# Preencher os valores inconsistente com a média das idades
# print(var_tblCredit.mean())

# print(var_tblCredit['person_age'].mean())

var_tblCredit['person_age'][var_tblCredit['person_age'] > 0].mean()

# var_tblCredit.loc[var_tblCredit['person_age'] < 0, 'person_age'] = 40.92

# print(var_tblCredit.isnull())
# print(var_tblCredit.isnull().sum())
var_tblTreino = var_tblCredit
var_tblTreino = var_tblTreino.drop(labels='person_home_ownership', axis=1)
var_tblTreino = var_tblTreino.drop(labels='person_emp_length', axis=1)
var_tblTreino = var_tblTreino.drop(labels='loan_intent', axis=1)
var_tblTreino = var_tblTreino.drop(labels='loan_grade', axis=1)
var_tblTreino = var_tblTreino.drop(labels='loan_int_rate', axis=1)
var_tblTreino = var_tblTreino.drop(labels='loan_status', axis=1)
var_tblTreino = var_tblTreino.drop(labels='loan_percent_income', axis=1)
var_tblTreino = var_tblTreino.drop(labels='cb_person_cred_hist_length', axis=1)
# print(var_tblTreino.isnull().sum())

# print(var_tblTreino.loc(pd.isnull(var_tblTreino['person_age'])))
var_tblTreino['person_age'].fillna(var_tblTreino['person_age'].mean(), inplace=True)

var_tblXCredit = var_tblTreino.iloc[:, 1:4].values
# print(var_tblXCredit)

var_tblYCredit = var_tblTreino.iloc[:, :4].values
# print(var_tblYCredit)

# print(var_tblXCredit[:, 0])

var_tblXCredit = var_objScalerCredit.fit_transform(var_tblXCredit)