
#Creamos 50 grafos de 25 nodos cada uno con 60 aristas(20% densidad)
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

diccionario_grafos=dict()
for i in range(50):
    G = nx.Graph()
    num_nodos = 25
    nodos = range(num_nodos)
    G.add_nodes_from(nodos)
    num_aristas = 60
    while len(list(G.edges())) < num_aristas:
        nodo1 = random.choice(nodos)
        nodo2 = random.choice(nodos)
        if nodo1 != nodo2:
            if (nodo1,nodo2) not in list(G.edges()) and (nodo2,nodo1) not in list(G.edges()):
                G.add_edge(nodo1, nodo2)
    clave= 'G_'+ str(i+1)
    diccionario_grafos[clave]=G


nx.write_gpickle(diccionario_grafos, "grafos_25_20.gpickle")
print(diccionario_grafos)



#Algoritmo Louvain

import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

grafos=nx.read_gpickle("grafos_25_20.gpickle")
tiempos_computacion_louvain=dict() 
resultados_louvain=dict()
p=1
for grafo in list(grafos.values()):
    for arista in grafo.edges():
        grafo.edges[arista[0], arista[1]]['weight']=1   
    
    inicio = time.time()
    resultado=nx.community.louvain_communities(grafo,seed=3)
    tiempo_transcurrido = time.time() - inicio
    clave_="Tiempo de computación algoritmo Louvain grafo"+ str(p)
    tiempos_computacion_louvain[clave_]=tiempo_transcurrido
    clave2="Partición Louvain grafo"+str(p)
    resultados_louvain[clave2]=resultado
    p=p+1
print (tiempos_computacion_louvain)
print(resultados_louvain)
nx.write_gpickle(tiempos_computacion_louvain, "Tiempos_computacion_louvain_25_20_densidad.gpickle")
nx.write_gpickle(resultados_louvain, "comunidades_louvain_25_20_densidad.gpickle")

comunidades = nx.read_gpickle("comunidades_louvain_25_20_densidad.gpickle")

q=1
modularidades_louvain=dict()
for particion in list(comunidades.values()):
    clave_grafo= 'G_'+str(q)
    grafo_actual=grafos[clave_grafo]
    modularidad=nx.community.modularity(grafo_actual, particion)
    clave_m= 'Modularidad Louvain del grafo'+str(q)
    modularidades_louvain[clave_m]=modularidad
    q=q+1

print(modularidades_louvain)    
nx.write_gpickle(modularidades_louvain, "Modularidades_louvain_25_20_densidad.gpickle")    





#Algoritmo kerninganlin iterado

def repeticion_KernighanLin(grafo):
    random.seed(60)
    particion_en_dos= nx.community.kernighan_lin_bisection(grafo)
    anterior_maxima_modularidad=  nx.community.modularity (grafo,particion_en_dos)
    particion=list(particion_en_dos)
    stop=0
    while stop==0:
        stop=1
        posibles_particiones=dict()
        for comunidad in particion:
            nueva_particion=particion.copy()
            subgrafo=grafo.subgraph(list(comunidad))
            random.seed(60)
            nuevas_comunidades=nx.community.kernighan_lin_bisection(subgrafo)
            nueva_particion.remove(comunidad)
            nueva_particion.extend(list(nuevas_comunidades))   
            modularity= nx.community.modularity(grafo,nueva_particion) 
            posibles_particiones[modularity]= nueva_particion
        maxima_modularidad= max(list(posibles_particiones.keys()))
        if maxima_modularidad>anterior_maxima_modularidad:
            particion=posibles_particiones[maxima_modularidad]
            anterior_maxima_modularidad=maxima_modularidad
            stop=0

        else:
            stop=1
    
    return particion

grafos=nx.read_gpickle("grafos_25_20.gpickle")
tiempos_computacion_kl=dict() 
resultados_kl=dict()
p=1
for grafo in list(grafos.values()):
    
    for arista in grafo.edges():
        grafo.edges[arista[0], arista[1]]['weight']=1   
    inicio = time.time()
    resultado= repeticion_KernighanLin(grafo)
    tiempo_transcurrido = time.time() - inicio
    clave_="Tiempo de computación algoritmo reiterado Kernighan-Lin grafo"+ str(p)
    tiempos_computacion_kl[clave_]=tiempo_transcurrido
    clave2="Partición reiterado Kernighan-Lin grafo"+str(p)
    resultados_kl[clave2]=resultado
    p=p+1
print (tiempos_computacion_kl)
print(resultados_kl)
nx.write_gpickle(tiempos_computacion_kl, "Tiempos_computacion_reiterado_KL_25_20_densidad.gpickle")
nx.write_gpickle(resultados_kl, "comunidades_reiterado_KL_25_20_densidad.gpickle")

comunidades = nx.read_gpickle("comunidades_reiterado_KL_25_20_densidad.gpickle")

q=1
modularidades_kl=dict()
for particion in list(comunidades.values()):
    clave_grafo= 'G_'+str(q)
    grafo_actual=grafos[clave_grafo]
    modularidad=nx.community.modularity(grafo_actual,  particion)
    clave_m= 'Modularidad reiterado KL del grafo'+str(q)
    modularidades_kl[clave_m]=modularidad
    q=q+1

print(modularidades_kl)    
nx.write_gpickle(modularidades_kl, "Modularidades_reiterado_KL_25_20_densidad.gpickle")    




#Algoritmo Girvan y Newman
import numpy as np
import networkx as nx
import community

def max_edge_betweennes(edge_betweenness):
    maxim=0
    edge_elegido=None
    for edge in edge_betweenness.keys():
        if edge_betweenness[edge]>maxim:
            edge_elegido=edge
            maxim=edge_betweenness[edge]
            
    return edge_elegido , maxim 

def Girvan_Newman(grafo):
    grafo_copia=grafo.copy()
    particiones=[]
    particion_actual=[]
    while list(grafo_copia.edges())!=[]:
        comunidades=None
        edge_betweenness=nx.edge_betweenness_centrality(grafo_copia)
        edge_remove=max_edge_betweennes(edge_betweenness)[0]
        grafo_copia.remove_edge(edge_remove[0],edge_remove[1])
        componentes=list(nx.connected_components(grafo_copia))
        if len(componentes) > len (particion_actual):
            particion_actual= componentes
            particiones.append(componentes)
    
    diccionario_particiones_modularidad=modularidad_cada_particion(particiones,grafo)
    particion_elegida=particion_mas_modularidad( diccionario_particiones_modularidad)
   
    return particion_elegida



def modularidad_cada_particion(distintas_particiones,grafo):
    diccionario=dict()
    for particion in distintas_particiones:
        modularidad=nx.community.modularity(grafo,particion)
        diccionario[modularidad]=particion
    
    return diccionario       
        

def particion_mas_modularidad(diccionario):
    max_modularidad=max(diccionario.keys())
    elegida=diccionario[max_modularidad]
    return elegida



grafos=nx.read_gpickle("grafos_25_20.gpickle")
tiempos_computacion_GN=dict() 
resultados_GN=dict()
p=1
for grafo in list(grafos.values()):
    
    for arista in grafo.edges():
        grafo.edges[arista[0], arista[1]]['weight']=1   
    inicio = time.time()
    resultado= Girvan_Newman(grafo)
    tiempo_transcurrido = time.time() - inicio
    clave_="Tiempo de computación algoritmo Girvan-Newman grafo"+ str(p)
    tiempos_computacion_GN[clave_]=tiempo_transcurrido
    clave2="Partición Girvan-Newman grafo"+str(p)
    resultados_GN[clave2]=resultado
    p=p+1
print (tiempos_computacion_GN)
print(resultados_GN)
nx.write_gpickle(tiempos_computacion_GN, "Tiempos_computacion_Girvan-Newman_25_20_densidad.gpickle")
nx.write_gpickle(resultados_GN, "comunidades_Girvan-Newman_25_20_densidad.gpickle")

comunidades = nx.read_gpickle("comunidades_Girvan-Newman_25_20_densidad.gpickle")

q=1
modularidades_GN=dict()
for particion in list(comunidades.values()):
    clave_grafo= 'G_'+str(q)
    grafo_actual=grafos[clave_grafo]
    modularidad=nx.community.modularity(grafo_actual,  particion)
    clave_m= 'Modularidad Girvan-Newman del grafo'+str(q)
    modularidades_GN[clave_m]=modularidad
    q=q+1

print(modularidades_GN)    
nx.write_gpickle(modularidades_GN, "Modularidades_Girvan-Newman_25_20_densidad.gpickle")    




#Algoritmo GA-Net
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.core.debugger import set_trace as breakpoint
import time



def create_individuo(grafo):
    individuo=dict()
    for nodo in grafo.nodes():
        if list(grafo.neighbors(nodo))!=[]:
            gen=random.choice(list(grafo.neighbors(nodo)))
            individuo[nodo]=gen
        else:
            individuo[nodo]=nodo
    return individuo    


def primera_generacion(num_individuos,grafo):
    generacion_1 = []
    while len(generacion_1)< num_individuos :
        individuo = create_individuo(grafo)
        if individuo not in generacion_1:
            generacion_1.append(individuo)
    return generacion_1

def seleccionar_padres(poblacion, evaluaciones):
    transformacion=[]
    for evaluacion in evaluaciones:
        transformacion.append(np.exp(evaluacion))
    suma_total=0
    for elemento in transformacion:
        suma_total=suma_total + elemento
    probabilidades = []
    for elemento in transformacion:
        probabilidades.append(elemento/suma_total)
        
    padres_seleccionados = random.choices(poblacion, weights=probabilidades, k=2)

    return padres_seleccionados

def seleccionar_individuos(poblacion, evaluaciones,n):
    transformacion=[]
    for evaluacion in evaluaciones:
        transformacion.append(np.exp(evaluacion))
    suma_total=0
    for elemento in transformacion:
        suma_total=suma_total + elemento
    probabilidades = []
    for elemento in transformacion:
        probabilidades.append(elemento/suma_total)
        
    individuos_seleccionados = random.choices(poblacion, weights=probabilidades, k=2)

    return individuos_seleccionados

def reproduccion(padre1, padre2):
    hijo = dict()
    codific_binaria=random.choices([0,1],weights=[0.5,0.5],k=len(padre1))
    for i in range(len(padre1)):
        if codific_binaria[i]==1:
            hijo[i]=padre1[i]
        elif codific_binaria[i]==0:
            hijo[i]=padre2[i]
            
    return hijo

def mutacion(individuo, probabilidad_mutacion,grafo):
    individuo_mutado = individuo.copy()
    for i in sorted(list(grafo.nodes())):
        n=random.uniform(0,1)
        gen = individuo[i]
        if n <= probabilidad_mutacion: 
            if list(grafo.neighbors(i))!=[]:
                vecinos= grafo.neighbors(i)
                individuo_mutado[i] = random.choice(list(vecinos))
            else:
                individuo_mutado[i]=i
    return individuo_mutado


def genetic_algorithm(grafo,num_individuos, num_generaciones,probabilidad_mutacion,crossover_rate):
    poblacion= primera_generacion(num_individuos,grafo)
    best_fitness = float('-inf')
    anterior_max_fitness=None
    mejor_individuo = None
    mejores_individuos_generaciones=[]
    c=0
    for generacion in range(num_generaciones-1):
        individuos_evaluados = dict()
        for individuo in poblacion:
            de_individuo_a_comunidad=individuo_comunidades(individuo)
            lista_de_sets = [set(lista) for lista in de_individuo_a_comunidad]
            fitness =nx.community.modularity(grafo,lista_de_sets)
            individuos_evaluados[fitness]=individuo
            
        max_fitness= max(individuos_evaluados.keys())
        if max_fitness >= best_fitness:
                best_fitness =max_fitness
                mejor_individuo = individuos_evaluados[max_fitness]
                generacion_mejor_individuo= generacion +1
            
        else:
            c=c+1
                
        siguiente_generacion = []
        while len(siguiente_generacion) < num_individuos:
            progenitor1, progenitor2 = seleccionar_padres(list(individuos_evaluados.values()), list(individuos_evaluados.keys()))
            if random.uniform(0,1) < crossover_rate:
                hijo = reproduccion(progenitor1, progenitor2)
                mutacion_hijo = mutacion(hijo, probabilidad_mutacion,grafo)
                siguiente_generacion.append(mutacion_hijo)
            else:
                siguiente_generacion.append(dict())
        
        
        contador_vacios = len([vacio for vacio in siguiente_generacion if not vacio])
        siguiente_generacion = [d for d in siguiente_generacion if d]
        individuos_no_mutados= seleccionar_individuos(list(individuos_evaluados.values()), list(individuos_evaluados.keys()),contador_vacios)
        siguiente_generacion.extend(list(individuos_no_mutados))
        poblacion = siguiente_generacion
    
    
    while True:
        individuos_evaluados = dict()
        for individuo in poblacion:
            de_individuo_a_comunidad=individuo_comunidades(individuo)
            lista_de_sets = [set(lista) for lista in de_individuo_a_comunidad]
            fitness =nx.community.modularity(grafo,lista_de_sets)
            individuos_evaluados[fitness]=individuo
            
        max_fitness= max(individuos_evaluados.keys())
        if max_fitness >= best_fitness:
                best_fitness =max_fitness
                mejor_individuo = individuos_evaluados[max_fitness]
                generacion_mejor_individuo= generacion +1
                c=0
            
        else:
            c=c+1
        
        if c<5:    
            siguiente_generacion = []
            while len(siguiente_generacion) < num_individuos:
                progenitor1, progenitor2 = seleccionar_padres(list(individuos_evaluados.values()), list(individuos_evaluados.keys()))
                if random.uniform(0,1) < crossover_rate:
                    hijo = reproduccion(progenitor1, progenitor2)
                    mutacion_hijo = mutacion(hijo, probabilidad_mutacion,grafo)
                    siguiente_generacion.append(mutacion_hijo)
                else:
                    siguiente_generacion.append(dict())
        
        
            contador_vacios = len([vacio for vacio in siguiente_generacion if not vacio])
            siguiente_generacion = [d for d in siguiente_generacion if d]
            individuos_no_mutados= seleccionar_individuos(list(individuos_evaluados.values()), list(individuos_evaluados.keys()),contador_vacios)
            siguiente_generacion.extend(list(individuos_no_mutados))
            poblacion = siguiente_generacion
           
        else:
            break
        
    
    particion_mejor_individuo=individuo_comunidades(mejor_individuo)
    
    return mejor_individuo,particion_mejor_individuo,generacion_mejor_individuo,best_fitness 

def individuo_comunidades(individuo):
    comunidades=[]
    copia_individuo=individuo.copy()
    
    while copia_individuo!={}:
        nodo=list(copia_individuo.keys())[0]
        comunidad=[]
        comunidad_anterior=[]
        comunidad.append(nodo)
        if nodo!=copia_individuo[nodo]:
            comunidad.append(copia_individuo[nodo])
        
            while comunidad!=comunidad_anterior :
                comunidad_anterior=comunidad.copy()
                for clave in copia_individuo.keys():
                    for miembro in comunidad_anterior:
                        if copia_individuo[clave]==miembro and clave not in comunidad:
                            comunidad.append(clave)              
                             
        comunidades.append(comunidad)
        for elemento in comunidad:
            copia_individuo.pop(elemento, None)
            
    return comunidades


grafos=nx.read_gpickle("grafos_25_20.gpickle")
tiempos_computacion_GA=dict() 
resultados_GA=dict()
p=1
for grafo in list(grafos.values()):
    for arista in grafo.edges():
        grafo.edges[arista[0], arista[1]]['weight']=1   
    
    inicio = time.time()
    resultado=genetic_algorithm(grafo,300, 10,0.1,0.8)[1]
    tiempo_transcurrido = time.time() - inicio
    clave_="Tiempo de computación algoritmo GA-Net grafo"+ str(p)
    tiempos_computacion_GA[clave_]=tiempo_transcurrido
    clave2="Partición GA-Net grafo"+str(p)
    resultados_GA[clave2]=resultado
    p=p+1
print (tiempos_computacion_GA)
print(resultados_GA)
nx.write_gpickle(tiempos_computacion_GA, "Tiempos_computacion_GA-Net_25_20_densidad.gpickle")
nx.write_gpickle(resultados_GA, "comunidades_GA-Net_25_20_densidad.gpickle")

comunidades = nx.read_gpickle("comunidades_GA-Net_25_20_densidad.gpickle")

q=1
modularidades_GA=dict()
for particion in list(comunidades.values()):
    clave_grafo= 'G_'+str(q)
    grafo_actual=grafos[clave_grafo]
    modularidad=nx.community.modularity(grafo_actual,  particion)
    clave_m= 'Modularidad GA-Net del grafo'+str(q)
    modularidades_GA[clave_m]=modularidad
    q=q+1

print(modularidades_GA)    
nx.write_gpickle(modularidades_GA, "Modularidades_GA-Net_25_20_densidad.gpickle")    


#Boxplot tiempos de ejecución

import matplotlib.pyplot as plt
import networkx as nx

Tiempos_louvain=nx.read_gpickle("Tiempos_computacion_Louvain_25_20_densidad.gpickle")
Tiempos_KL=nx.read_gpickle("Tiempos_computacion_reiterado_KL_25_20_densidad.gpickle")
Tiempos_GN=nx.read_gpickle("Tiempos_computacion_Girvan-Newman_25_20_densidad.gpickle")
Tiempos_GA=nx.read_gpickle("Tiempos_computacion_GA-Net_25_20_densidad.gpickle")



tiempos_algoritmos = {
    'Louvain': list(Tiempos_louvain.values()),
    'Kernighan-Lin':list(Tiempos_KL.values()),
    'Girvan-Newman':list(Tiempos_GN.values()),
    'GA-Net': list(Tiempos_GA.values())
}

datos = list(tiempos_algoritmos.values())

etiquetas = list(tiempos_algoritmos.keys())

bp=plt.boxplot(datos, labels=etiquetas, showmeans=True, meanline=True)
bp
plt.title('Boxplot de Tiempos de Ejecución grafo 25 nodos, 20% densidad de aristas')
plt.xlabel('Algoritmos')
plt.ylabel('Tiempo de Ejecución')


mediana = bp['medians'][0]

media = bp['means'][0]

labels = ['Medianas', 'Medias' ]

plt.legend([mediana , media], labels, loc='upper right')

plt.show()


#Algoritmo con mejor modularidad en cada partición
import networkx as nx
Modularidades_louvain=nx.read_gpickle("Modularidades_Louvain_25_20_densidad.gpickle")
Modularidades_KL=nx.read_gpickle("Modularidades_reiterado_KL_25_20_densidad.gpickle")
Modularidades_GN=nx.read_gpickle("Modularidades_Girvan-Newman_25_20_densidad.gpickle")
Modularidades_GA=nx.read_gpickle("Modularidades_GA-Net_25_20_densidad.gpickle")
grafos=nx.read_gpickle("grafos_25_20.gpickle")

primero_Louvain=0
primero_KL=0
primero_GN=0
primero_GA=0
t=0
for i in range(len(list(grafos.keys()))):
    lista_clave_valor=[]
    clave1='Modularidad Louvain del grafo'+str(i+1)
    clave2='Modularidad reiterado KL del grafo'+str(i+1)
    clave3='Modularidad Girvan-Newman del grafo'+str(i+1)
    clave4='Modularidad GA-Net del grafo'+str(i+1)
    lista_clave_valor.append((clave1,Modularidades_louvain[clave1]))
    lista_clave_valor.append((clave2,Modularidades_KL[clave2]))
    lista_clave_valor.append((clave3,Modularidades_GN[clave3]))
    lista_clave_valor.append((clave4,Modularidades_GA[clave4]))
    lista_ordenada = sorted(lista_clave_valor, key=lambda x: x[1],reverse=True)
    print(lista_ordenada)
    if lista_ordenada[0][0]==lista_ordenada[1][0]:
        t=t+1
    
    if lista_ordenada[0][0]==clave1:
        primero_Louvain=primero_Louvain+1
    elif lista_ordenada[0][0]==clave2:
        primero_KL=primero_KL+1
    elif lista_ordenada[0][0]==clave3:
        primero_GN=primero_GN+1
        print(lista_ordenada[0][0])
    elif lista_ordenada[0][0]==clave4:
        primero_GA=primero_GA+1 
        
print('Louvain'+' '+str(primero_Louvain))
print('KL'+' '+str(primero_KL))
print('GN'+' '+str(primero_GN))
print('GA'+' '+str(primero_GA))
print(t)



#Algoritmo peor modularidad encada partición
import networkx as nx
Modularidades_louvain=nx.read_gpickle("Modularidades_Louvain_25_20_densidad.gpickle")
Modularidades_KL=nx.read_gpickle("Modularidades_reiterado_KL_25_20_densidad.gpickle")
Modularidades_GN=nx.read_gpickle("Modularidades_Girvan-Newman_25_20_densidad.gpickle")
Modularidades_GA=nx.read_gpickle("Modularidades_GA-Net_25_20_densidad.gpickle")
grafos=nx.read_gpickle("grafos_25_20.gpickle")

ultimo_Louvain=0
ultimo_KL=0
ultimo_GN=0
ultimo_GA=0
t=0
for i in range(len(list(grafos.keys()))):
    lista_clave_valor=[]
    clave1='Modularidad Louvain del grafo'+str(i+1)
    clave2='Modularidad reiterado KL del grafo'+str(i+1)
    clave3='Modularidad Girvan-Newman del grafo'+str(i+1)
    clave4='Modularidad GA-Net del grafo'+str(i+1)
    lista_clave_valor.append((clave1,Modularidades_louvain[clave1]))
    lista_clave_valor.append((clave2,Modularidades_KL[clave2]))
    lista_clave_valor.append((clave3,Modularidades_GN[clave3]))
    lista_clave_valor.append((clave4,Modularidades_GA[clave4]))
    lista_ordenada = sorted(lista_clave_valor, key=lambda x: x[1],reverse=False)
    print(lista_ordenada)
    if lista_ordenada[0][0]==lista_ordenada[1][0]:
        t=t+1
    
    if lista_ordenada[0][0]==clave1:
        ultimo_Louvain=primero_Louvain+1
    elif lista_ordenada[0][0]==clave2:
        ultimo_KL=primero_KL+1
    elif lista_ordenada[0][0]==clave3:
        ultimo_GN=primero_GN+1
        print(lista_ordenada[0][0])
    elif lista_ordenada[0][0]==clave4:
        ultimo_GA=primero_GA+1 
        
print('Louvain'+' '+str(ultimo_Louvain))
print('KL'+' '+str(ultimo_KL))
print('GN'+' '+str(ultimo_GN))
print('GA'+' '+str(ultimo_GA))
print(t)



#Boxplot modularidades
import matplotlib.pyplot as plt
import networkx as nx
Modularidades_louvain=nx.read_gpickle("Modularidades_Louvain_25_20_densidad.gpickle")
Modularidades_KL=nx.read_gpickle("Modularidades_reiterado_KL_25_20_densidad.gpickle")
Modularidades_GN=nx.read_gpickle("Modularidades_Girvan-Newman_25_20_densidad.gpickle")
Modularidades_GA=nx.read_gpickle("Modularidades_GA-Net_25_20_densidad.gpickle")
grafos=nx.read_gpickle("grafos_25_20.gpickle")

Modularidades_algoritmos = {
    'Louvain': list(Modularidades_louvain.values()),
    'Kernighan-Lin':list(Modularidades_KL.values()),
    'Girvan-Newman':list(Modularidades_GN.values()),
    'GA-Net': list(Modularidades_GA.values())}

datos = list(Modularidades_algoritmos.values())

etiquetas = list(Modularidades_algoritmos.keys())

bp=plt.boxplot(datos, labels=etiquetas, showmeans=True, meanline=True)
bp
plt.title('Boxplot de Modularidades grafo 25 nodos, 20% densidad de aristas')
plt.xlabel('Algoritmos')
plt.ylabel('Modularidad')


mediana = bp['medians'][0]

media = bp['means'][0]

labels = ['Medianas', 'Medias' ]

plt.legend([mediana , media], labels, loc='upper right')

plt.show()



#Boxplot tamaños comunidades de cada algoritmo

Comunidades_louvain=nx.read_gpickle("comunidades_louvain_25_20_densidad.gpickle")
Comunidades_KL=nx.read_gpickle("comunidades_reiterado_KL_25_20_densidad.gpickle")
Comunidades_GN=nx.read_gpickle("comunidades_Girvan-Newman_25_20_densidad.gpickle")
Comunidades_GA=nx.read_gpickle("comunidades_GA-Net_25_20_densidad.gpickle")
grafos=nx.read_gpickle("grafos_25_20.gpickle")

numero_comunidades_louvain=dict()
numero_comunidades_KL=dict()
numero_comunidades_GN=dict()
numero_comunidades_GA=dict()

for i in range(len(list(grafos.keys()))):

    clave1='Cantidad comunidades Louvain del grafo'+str(i+1)
    clave2='Cantidad comunidades KL del grafo'+str(i+1)
    clave3='Cantidad comunidades Girvan-Newman del grafo'+str(i+1)
    clave4='Cantidad comunidades GA-Net del grafo'+str(i+1)
    
    numero_comunidades_louvain[clave1]= len(Comunidades_louvain['Partición Louvain grafo'+ str(i+1)])
    numero_comunidades_KL[clave2]= len(Comunidades_KL["Partición reiterado Kernighan-Lin grafo"+str(i+1)])
    numero_comunidades_GN[clave3]= len(Comunidades_GN["Partición Girvan-Newman grafo"+str(i+1)])
    numero_comunidades_GA[clave4]= len(Comunidades_GA["Partición GA-Net grafo"+str(i+1)])

    

Comunidades_algoritmos = {
    'Louvain': list(numero_comunidades_louvain.values()),
    'Kernighan-Lin':list(numero_comunidades_KL.values()),
    'Girvan-Newman':list(numero_comunidades_GN.values()),
    'GA-Net': list(numero_comunidades_GA.values())}

datos = list(Comunidades_algoritmos.values())

etiquetas = list(Comunidades_algoritmos.keys())

bp=plt.boxplot(datos, labels=etiquetas, showmeans=True, meanline=True)
bp
plt.title('Boxplot de cantidad de comunidades grafo 25 nodos, 20% densidad de aristas')
plt.xlabel('Algoritmos')
plt.ylabel('Cantidad comunidades')

mediana = bp['medians'][0]
print(mediana)

media = bp['means'][0]

labels = ['Medianas', 'Medias' ]

plt.legend([mediana , media], labels, loc='upper right')

plt.show()


print(numero_comunidades_GN)

#Diccionarios tamaños comunidades 

Comunidades_louvain=nx.read_gpickle("comunidades_louvain_25_20_densidad.gpickle")

def nodos_por_comunidad(lista_particiones):
    lista=[]
    for particion in lista_particiones:
        lista_nodos_comunidad=[]
        for comunidad in particion:
            lista_nodos_comunidad.append(len(comunidad))
        lista.append( lista_nodos_comunidad)
    return lista 
    
n_c=nodos_por_comunidad(list(Comunidades_louvain.values()))

def conteo_comunidades(lista):
    cantidades=dict()
    for n_nodos in lista:
        if n_nodos not in cantidades:
            cantidades[n_nodos]=1
        else:
            cantidades[n_nodos]+=1
    return cantidades 

i=1
for lista in n_c:
    diccionario=conteo_comunidades(lista)
    print('Número de comunidades de cada tamaño Louvain grafo G_'+str(i))  
    print(diccionario)
    i=i+1

    
    



Comunidades_KL=nx.read_gpickle("comunidades_reiterado_KL_25_20_densidad.gpickle")
def nodos_por_comunidad(lista_particiones):
    lista=[]
    for particion in lista_particiones:
        lista_nodos_comunidad=[]
        for comunidad in particion:
            lista_nodos_comunidad.append(len(comunidad))
        lista.append( lista_nodos_comunidad)
    return lista 
    
n_c=nodos_por_comunidad(list(Comunidades_KL.values()))

def conteo_comunidades(lista):
    cantidades=dict()
    for n_nodos in lista:
        if n_nodos not in cantidades:
            cantidades[n_nodos]=1
        else:
            cantidades[n_nodos]+=1
    return cantidades 

i=1
for lista in n_c:
    diccionario=conteo_comunidades(lista)
    print('Número de comunidades de cada tamaño Kernighan-Lin grafo G_'+str(i))  
    print(diccionario)
    i=i+1




Comunidades_GN=nx.read_gpickle("comunidades_Girvan-Newman_25_20_densidad.gpickle")
def nodos_por_comunidad(lista_particiones):
    lista=[]
    for particion in lista_particiones:
        lista_nodos_comunidad=[]
        for comunidad in particion:
            lista_nodos_comunidad.append(len(comunidad))
        lista.append( lista_nodos_comunidad)
    return lista 
    
n_c=nodos_por_comunidad(list(Comunidades_GN.values()))

def conteo_comunidades(lista):
    cantidades=dict()
    for n_nodos in lista:
        if n_nodos not in cantidades:
            cantidades[n_nodos]=1
        else:
            cantidades[n_nodos]+=1
    return cantidades 

i=1
for lista in n_c:
    diccionario=conteo_comunidades(lista)
    print('Número de comunidades de cada tamaño Girvan-Newman grafo G_'+str(i))  
    print(diccionario)
    i=i+1




Comunidades_GA=nx.read_gpickle("comunidades_GA-Net_25_20_densidad.gpickle")
def nodos_por_comunidad(lista_particiones):
    lista=[]
    for particion in lista_particiones:
        lista_nodos_comunidad=[]
        for comunidad in particion:
            lista_nodos_comunidad.append(len(comunidad))
        lista.append( lista_nodos_comunidad)
    return lista 
    
n_c=nodos_por_comunidad(list(Comunidades_GA.values()))

def conteo_comunidades(lista):
    cantidades=dict()
    for n_nodos in lista:
        if n_nodos not in cantidades:
            cantidades[n_nodos]=1
        else:
            cantidades[n_nodos]+=1
    return cantidades 

i=1
for lista in n_c:
    diccionario=conteo_comunidades(lista)
    print('Número de comunidades de cada tamaño GA-Net grafo G_'+str(i))  
    print(diccionario)
    i=i+1

