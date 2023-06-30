# Codigo-TFG-deteccion-de-comunidades
En este repositorio se incluye el código desarrollado para el caso práctico de Twitter del TFG. Así mismo, se encuentra otro archivo que corresponde al código escrito
para la comparación de los cuatro algoritmos explicados en el trabajo en el caso de grafos de 25 nodos y 20% de densidad de aristas. El código para la comparación
de los algoritmos en el resto de casos planteados en el trabajo es idéntico, lo único que cambia es el número de nodos y aristas definidos al inicio del archivo a la
hora de generar el conjunto de grafos que va a ser estudiado. 
También se encuentran en este repositorio los archivos utilizados y generados en el caso práctico. Los archivos de la comparación de algoritmos, están todos en formato 
gpickle y cada uno de ellos contiene la información descrita en el nombre del propio archivo. Para abrir dicho algoritmo y acceder a su contenido basta escribir lo siguienete
en un archivo de Python y ejecutarlo:       
diccionario = nx.read_gpickle("Nombre_del_archivo.gpickle")                                                                                
print(diccionario)                                                                                                                      
Los archivos correspondientes gpickle del caso de Twitter contienen listas y para ser visualizadas ha de hacerse los mismo que en las dos líneas anteriores.
En la carpeta donde se encuentran los archivos del caso de Twitter se incluye el csv de los "tweets" utilizados para crear el grafo.
