En primer lugar, una vez obtenido el archivo .csv (mediante la herramienta tshark.exe) de cada escenario, se obtiene:
  escenario_normal.csv
  escenario_dos.csv
  escenario_ddos.csv
  escenario_mitm.csv (Este ultimo aun falta la recoleccion)
Luego, se adiciona la columna 'escenario' para denominarla como la variable a predecir, toma los siguientes valores:
  0 = normal
	1 = DDoS
	2 = DoS
  3 = MitM (En proceso)
En segundo lugar, se unen los datasets (tomando 100k de cada uno); posteriormete,se eliminan las columnas que son redundantes o no necesarias 'mqtt.willtopic', 'mqtt.willmsg', 'mqtt.conflags', 
'mqtt.conack.flags' y 'mqtt.hdrflags', estas dos ultimas primerso se dividen para luego ser eliminadas. Despues, se aplican tecnicas de encoding  y normalizacion.
En tercer lugar, se aplica los algoritmos Pearson coefficient correlation (PCC), ExtraTreesClassifier y RandomForestClassifier, para seleccionar las caracteristicas mas relevantes.
Posteriormente, se crea un dataset con los algoritmos seleccionados, para usarlo en el modelo.
En cuarto lugar, se divide el dataset en 80 - 20, en la primera porcion se usa la tecnica de validacion cruzada con 10 iteraciones para obtener los mejores hiperparametros para cada algoritmo y entrenarlo.
Finalmente, se crea el modelo mediante la tecnica de votacion, validandolo con los demas datos.
