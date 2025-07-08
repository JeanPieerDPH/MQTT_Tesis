import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Cargar tu dataset (ajusta el nombre del archivo)
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\3_dataset_final_sin_depu.csv')

df2 = [
'_ws.col.protocol_MQTT',
'_ws.col.protocol_TCP',

'frame.time_epoch',
'frame.len',
'frame.time_relative',
'frame.time_delta_displayed',

'tcp.time_relative',
'tcp.stream',

'tcp.hdr_len',
'tcp.ACK',
'tcp.window_size',
'tcp.srcport',
'tcp.SYN',
'tcp.dstport',
'tcp.len',
'tcp.analysis.acks_frame',
'escenario']
#estes con los 4 escenarios
df3= [
    '_ws.col.protocol_ARP',
    '_ws.col.protocol_TCP',
    '_ws.col.protocol_MQTT',

    'frame.time_epoch',
    'frame.time_delta_displayed',
    'frame.time_relative',
    'frame.len',

    'tcp.hdr_len',
    'tcp.time_relative',
    'tcp.ACK',
    'tcp.dstport',
    'tcp.srcport',
    'tcp.analysis.acks_frame',
    'tcp.stream',
    'tcp.SYN',
    'tcp.window_size',
    'escenario']

dataset_recortado = df[df3]
dataset_recortado.to_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\4_dataset_final_recortado_sin_depu.csv', index=False)

