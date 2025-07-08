import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Simulación de datos (puedes cargar desde .csv también)
df = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\2_combinado_400k_sin_depu.csv')


# Inicializar el scaler
scaler = MinMaxScaler()

# NOTA
# Antes de unir el dataset, se agrego la columna 'escenario' a cada dataset, para lograr identificar el ataque.
# Posteriormente se unieron los datasets.

# Filtrar las filas donde el valor en _ws.col.protocol no sea '0x0e00', 'BROWSER', 'MDNS', 'UDP/XML', 'ICMPv6', 'ICMP', 'IPv4', 'SSDP'
#df = df[~df['_ws.col.protocol'].isin(['0x0e00', 'BROWSER', 'MDNS', 'UDP/XML', 'ICMPv6', 'ICMP', 'IPv4', 'SSDP','NBNS','LLMNR','IGMPv3'])]

# Aplicar One-Hot Encoding a la columna '_ws.col.protocol'
df = pd.get_dummies(df, columns=['_ws.col.protocol'])
df['_ws.col.protocol_MQTT'] = df['_ws.col.protocol_MQTT'].astype(int)
df['_ws.col.protocol_TCP'] = df['_ws.col.protocol_TCP'].astype(int)
df['_ws.col.protocol_ARP'] = df['_ws.col.protocol_ARP'].astype(int)

#eliminar columnas, que son redundantes o no necesarias
df.drop(columns=['mqtt.willtopic'], inplace=True)
df.drop(columns=['mqtt.willmsg'], inplace=True)
df.drop(columns=['mqtt.conflags'], inplace=True)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Dividir la columna 'tcp.stream'
df['tcp.stream'] = df['tcp.stream'].fillna(0) # Reemplaza NaN con 0
df['tcp.stream'] = scaler.fit_transform(df[['tcp.stream']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Formato string antes de reemplazar
df['tcp.options.nop'] = df['tcp.options.nop'].astype(str)  # Convertir todo a texto
df['tcp.options.nop'] = df['tcp.options.nop'].apply(lambda x: 1 if x == '01' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.hdr_len'
df['tcp.hdr_len'] = df['tcp.hdr_len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.hdr_len'] = scaler.fit_transform(df[['tcp.hdr_len']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.analysis.acks_frame'
df['tcp.analysis.acks_frame'] = df['tcp.analysis.acks_frame'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.analysis.acks_frame'] = scaler.fit_transform(df[['tcp.analysis.acks_frame']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['tcp.checksum.status'] = df['tcp.checksum.status'].fillna(2) # Reemplaza NaN con 2
df['tcp.checksum.status'] = df['tcp.checksum.status'].apply(lambda x: 1 if x == 2 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.window_size'
df['tcp.window_size'] = df['tcp.window_size'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.window_size'] = scaler.fit_transform(df[['tcp.window_size']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.time_relative'
df['tcp.time_relative'] = df['tcp.time_relative'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.time_relative'] = scaler.fit_transform(df[['tcp.time_relative']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.analysis.initial_rtt'
df['tcp.analysis.initial_rtt'] = df['tcp.analysis.initial_rtt'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.analysis.initial_rtt'] = scaler.fit_transform(df[['tcp.analysis.initial_rtt']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.srcport'
df['tcp.srcport'] = df['tcp.srcport'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.srcport'] = scaler.fit_transform(df[['tcp.srcport']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.dstport'
df['tcp.dstport'] = df['tcp.dstport'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.dstport'] = scaler.fit_transform(df[['tcp.dstport']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.time_delta'
df['tcp.time_delta'] = df['tcp.time_delta'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.time_delta'] = scaler.fit_transform(df[['tcp.time_delta']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'tcp.len'
df['tcp.len'] = df['tcp.len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['tcp.len'] = scaler.fit_transform(df[['tcp.len']])

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Dividir la columna 'tcp.flags'
def expand_flags(hex_flag):
    try:
        dec = int(hex_flag, 16)
        bin_str = format(dec, '08b')
        return pd.Series(
            [int(b) for b in bin_str[::-1]],
            index=['tcp.FIN', 'tcp.SYN', 'tcp.RST', 'tcp.PSH', 'tcp.ACK', 'tcp.URG', 'tcp.ECE', 'tcp.CWR']
        )
    except:
        return pd.Series([None]*8, index=['tcp.FIN', 'tcp.SYN', 'tcp.RST', 'tcp.PSH', 'tcp.ACK', 'tcp.URG', 'tcp.ECE', 'tcp.CWR'])

df_flags = df['tcp.flags'].apply(expand_flags)
df = pd.concat([df.drop(columns=['tcp.flags']), df_flags], axis=1)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'frame.time_relative'
df['frame.time_relative'] = df['frame.time_relative'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['frame.time_relative'] = scaler.fit_transform(df[['frame.time_relative']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'frame.time_epoch'
df['frame.time_epoch'] = df['frame.time_epoch'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['frame.time_epoch'] = scaler.fit_transform(df[['frame.time_epoch']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'frame.time_delta_displayed'
df['frame.time_delta_displayed'] = df['frame.time_delta_displayed'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['frame.time_delta_displayed'] = scaler.fit_transform(df[['frame.time_delta_displayed']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'frame.len'
df['frame.len'] = df['frame.len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['frame.len'] = scaler.fit_transform(df[['frame.len']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'mqtt.topic_len'
df['mqtt.topic_len'] = df['mqtt.topic_len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.topic_len'] = scaler.fit_transform(df[['mqtt.topic_len']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Rellenar la columna 'mqtt.willtopic_len'
df['mqtt.willtopic_len'] = df['mqtt.willtopic_len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.willtopic_len'] = scaler.fit_transform(df[['mqtt.willtopic_len']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Eliminar la columna 'mqtt.willtopic'

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Aplicar el escalado SOLO a la columna 'mqtt.willmsg_len'
df['mqtt.willmsg_len'] = df['mqtt.willmsg_len'].fillna(0) # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.willmsg_len'] = scaler.fit_transform(df[['mqtt.willmsg_len']]).ravel()
# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Eliminar la columna 'mqtt.willmsg'

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.ver'] = df['mqtt.ver'].fillna(4) # Reemplaza NaN con 4
df['mqtt.ver'] = df['mqtt.ver'].apply(lambda x: 1 if x == 4 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.suback.qos'] = df['mqtt.suback.qos'].fillna(0) # Reemplaza NaN con 0
df['mqtt.suback.qos'] = df['mqtt.suback.qos'].apply(lambda x: 1 if x == 0 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.sub.qos'] = df['mqtt.sub.qos'].fillna(0)  # Reemplaza NaN con 0
df['mqtt.sub.qos'] = df['mqtt.sub.qos'].apply(lambda x: 1 if x == 0 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.retain'] = df['mqtt.retain'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.qos'] = df['mqtt.qos'].fillna(0)  # Reemplaza NaN con 0
df['mqtt.qos'] = df['mqtt.qos'].apply(lambda x: 1 if x == 0 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.protoname'] = df['mqtt.protoname'].fillna('MQTT')  # Reemplaza NaN con texto MQTT
df['mqtt.protoname'] = df['mqtt.protoname'].apply(lambda x: 1 if x == 'MQTT' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.proto_len'] = df['mqtt.proto_len'].fillna(4)  # Reemplaza NaN con 4
df['mqtt.proto_len'] = df['mqtt.proto_len'].apply(lambda x: 1 if x == 4 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'mqtt.msgtype'
df['mqtt.msgtype'] = df['mqtt.msgtype'].fillna(0)  # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.msgtype'] = scaler.fit_transform(df[['mqtt.msgtype']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'mqtt.msgid'
df['mqtt.msgid'] = df['mqtt.msgid'].fillna(0)  # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.msgid'] = scaler.fit_transform(df[['mqtt.msgid']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'mqtt.msg'
def hex_to_float(hex_str):
    try:
        text = bytes.fromhex(hex_str).decode('ascii')
        return float(text)
    except Exception:
        return 0  # Si hay algún error, devolver NaN

df['mqtt.msg'] = df['mqtt.msg'].apply(hex_to_float)

# Escalar
scaler = MinMaxScaler()
df['mqtt.msg'] = scaler.fit_transform(df[['mqtt.msg']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Normalizar la columna 'mqtt.len'
df['mqtt.len'] = df['mqtt.len'].fillna(0)  # Reemplaza NaN con 0
scaler = MinMaxScaler()  # Re-inicializar el scaler para evitar problemas de ajuste
df['mqtt.len'] = scaler.fit_transform(df[['mqtt.len']]).ravel()

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.kalive'] = df['mqtt.kalive'].fillna(15)  # Reemplaza NaN con texto vacío
df['mqtt.kalive'] = df['mqtt.kalive'].apply(lambda x: 1 if x == 15 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Descomponer valores hexadecimales en bits
df['mqtt.hdrflags'] = df['mqtt.hdrflags'].apply(lambda x: int(str(x), 16) if pd.notnull(x) else 0)

# Extraer bits de flags (4 bits menos significativos)
df['mqtt.qos_bit1'] = df['mqtt.hdrflags'].apply(lambda x: (x >> 2) & 0x01)
df['mqtt.qos_bit0'] = df['mqtt.hdrflags'].apply(lambda x: (x >> 1) & 0x01)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.dupflag'] = df['mqtt.dupflag'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Aplicar el eliminado SOLO a la columna 'mqtt.conflags'

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.willflag'] = df['mqtt.conflag.willflag'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.uname'] = df['mqtt.conflag.uname'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.retain'] = df['mqtt.conflag.retain'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.reserved'] = df['mqtt.conflag.reserved'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.conflag.qos'] = df['mqtt.conflag.qos'].fillna(0)  # Reemplaza NaN con 0
df['mqtt.conflag.qos'] = df['mqtt.conflag.qos'].apply(lambda x: 1 if x == 0 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.passwd'] = df['mqtt.conflag.passwd'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conflag.cleansess'] = df['mqtt.conflag.cleansess'].apply(lambda x: 1 if str(x).upper() == 'FALSE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores
df['mqtt.conack.val'] = df['mqtt.conack.val'].fillna(0)  # Reemplaza NaN con 0
df['mqtt.conack.val'] = df['mqtt.conack.val'].apply(lambda x: 1 if x == 0 else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conack.flags.sp'] = df['mqtt.conack.flags.sp'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Reemplazar valores por 0 (normal), si hay otro dato es raro
df['mqtt.conack.flags.reserved'] = df['mqtt.conack.flags.reserved'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-
# Aplicar el eliminado SOLO a la columna 'mqtt.conack.flags'

# -*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*--*-*-*-*-*-*-



df.drop(columns=['mqtt.conack.flags'], inplace=True)
df.drop(columns=['mqtt.hdrflags'], inplace=True)

# O rellenarlos con 0 o la media
df = df.fillna(0)

df.to_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\3_dataset_final_sin_depu.csv', index=False)
