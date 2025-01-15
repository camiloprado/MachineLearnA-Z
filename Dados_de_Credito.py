import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import kagglehub, os

# Download latest version
var_strPath = kagglehub.dataset_download(handle="laotse/credit-risk-dataset")
var_strFileName = "credit_risk_dataset.csv"
var_strPathCredit = os.path.join(var_strPath, var_strFileName)

var_tblCredit = pd.read_csv(var_strPathCredit)