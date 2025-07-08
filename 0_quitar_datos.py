import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Simulación de datos (puedes cargar desde .csv también)
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\0_escenario_mitm.csv')


# Inicializar el scaler
scaler = MinMaxScaler()

# NOTA
# Antes de unir el dataset, se agrego la columna 'escenario' a cada dataset, para lograr identificar el ataque.
# Posteriormente se unieron los datasets.

# Filtrar las filas donde el valor en _ws.col.protocol no sea '0x0e00', 'BROWSER', 'MDNS', 'UDP/XML', 'ICMPv6', 'ICMP', 'IPv4', 'SSDP'
df = df[~df['_ws.col.protocol'].isin(['0x0e00', 'BROWSER', 'MDNS', 'UDP/XML', 'ICMPv6', 'ICMP', 'IPv4', 'SSDP','NBNS','LLMNR','IGMPv3'])]


df.to_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\1_escenario_mitm_protocolos_listos.csv', index=False)
