import pandas as pd

# Cargar los primeros 100000 datos de cada archivo
df1 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\escenario_normal_completo.csv', nrows=100000)
df2 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\escenario_ddos_completo.csv', nrows=100000)
df3 = pd.read_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\escenario_dos_completo.csv', nrows=100000)

# Unir los tres DataFrames
df_combinado = pd.concat([df1, df2, df3], ignore_index=True)

# Verificar tamaño final
print(df_combinado.shape)  # Debería mostrar (300000, columnas)

# Guardar en un nuevo archivo
df_combinado.to_csv('C:\\Users\\delpo\\Documents\\1_final\\Universidad\\Z_Tesis\\Topologia\\Data\\combinado_300k.csv', index=False)
