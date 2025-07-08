import pandas as pd

# Cargar los primeros 100000 datos de cada archivo
#df1 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\1_escenario_normal_protocolos_listos.csv', nrows=100000)
#df2 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\1_escenario_ddos_protocolos_listos.csv', nrows=100000)
#df3 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\1_escenario_dos_protocolos_listos.csv', nrows=100000)
#df4 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\1_escenario_mitm_protocolos_listos.csv', nrows=100000)

df1 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\0_escenario_normal.csv', nrows=100000)
df2 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\0_escenario_ddos.csv', nrows=100000)
df3 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\0_escenario_dos.csv', nrows=100000)
df4 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\0_escenario_mitm.csv', nrows=100000)

# Unir los tres DataFrames
df_combinado = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Verificar tamaño final
print(df_combinado.shape)  # Debería mostrar (300000, columnas)

# Guardar en un nuevo archivo
df_combinado.to_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\final\\2_combinado_400k_sin_depu.csv', index=False)
