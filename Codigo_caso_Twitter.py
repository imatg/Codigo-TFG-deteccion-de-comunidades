#!/usr/bin/env python
# coding: utf-8

# In[ ]:


modularidad_L=nx.community.modularity(G,comunidades_louvain)
numero_comunidades_L=len(comunidades_louvain)
print('La modularidad de la partición es'+ ' '+str(modularidad_L))
print('El número de comunidades de la partición es'+' '+str(numero_comunidades_L))


with open('modularidad_Louvain.pickle', 'wb') as archivo:
    pickle.dump(modularidad_L, archivo)
with open('Número_comunidades_Louvain.pickle', 'wb') as archivo:
    pickle.dump(numero_comunidades_L, archivo)


def comunidades_diccionario(comunidades):
    diccionario_comunidades=dict()
    for i in range(len(comunidades)):
        for nodo in comunidades[i]:
            diccionario_comunidades[nodo]=i
    return diccionario_comunidades


#Dibujamos las comunidades de Louvain
def dibujo_comunidades_louvain(G, comunidades):
    pos = nx.random_layout(G,seed=5)
    cmap = cm.get_cmap('viridis', max(comunidades.values()) + 1)
    nx.draw_networkx_nodes(G, pos, comunidades.keys(), node_size=8,
    cmap=cmap, node_color=list(comunidades.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

dibujo_louvain= dibujo_comunidades_louvain(G, comunidades_diccionario(comunidades_louvain))
      


#Algoritmo de Girvan y Newman
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
inicio=time.time()
comunidades_GN=Girvan_Newman(G)
Tiempo_computacion_GN=time.time()-inicio
print('Tiempo ejecución Girvan-Newman'+' '+str(Tiempo_computacion_GN))
print('La partición encontrada por Girvan-Newman es ')   
print(comunidades_GN)



with open('comunidades_G-N.pickle', 'wb') as archivo:
    pickle.dump(comunidades_GN, archivo)
with open('Tiempo_G-N.pickle', 'wb') as archivo:
    pickle.dump(Tiempo_computacion_GN, archivo)


with open('comunidades_G-N.pickle', 'rb') as archivo:
    comunidades_Girvan_Newman = pickle.load(archivo)

print( comunidades_Girvan_Newman)

with open('Tiempo_G-N.pickle', 'rb') as archivo:
    Tiempo_Girvan_Newman = pickle.load(archivo)

print( Tiempo_Girvan_Newman)


modularidad_GN=nx.community.modularity(G,comunidades_Girvan_Newman)
numero_comunidades_GN=len(comunidades_Girvan_Newman)
print('La modularidad de la partición es'+ ' '+str(modularidad_GN))
print('El número de comunidades de la partición es'+' '+str())


with open('modularidad_G-N.pickle', 'wb') as archivo:
    pickle.dump(modularidad_GN, archivo)
with open('Número_comunidades_G-N.pickle', 'wb') as archivo:
    pickle.dump(numero_comunidades_GN, archivo)

#Dibujo comunidades Girvan y Newman
def comunidades_diccionario(comunidades):
    diccionario_comunidades=dict()
    for i in range(len(comunidades)):
        for nodo in comunidades[i]:
            diccionario_comunidades[nodo]=i
    return diccionario_comunidades
print(len(set(list(comunidades_diccionario(comunidades_GN).values()))))

def dibujo_comunidades_GN(G, comunidades):
    pos = nx.random_layout(G,seed=5)
    cmap = cm.get_cmap('viridis', max(comunidades.values()) + 1)
    nx.draw_networkx_nodes(G, pos, comunidades.keys(), node_size=8,
    cmap=cmap, node_color=list(comunidades.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

dibujo_GN= dibujo_comunidades_GN(G, comunidades_diccionario(comunidades_GN))


#Algoritmo Kernighan-Lin iterado
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
            if len(comunidad)>1:
                nueva_particion=particion.copy()
                subgrafo=grafo.subgraph(list(comunidad))
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
            #print(particion)

        else:
            stop=1
    
    return particion
inicio=time.time()

comms_KL=repeticion_KernighanLin(G)
Tiempo_computacion_KL=time.time()-inicio
print('La partición encontrada por Kernighan Lin iterado es ')   
print(comms_KL)

print('El tiempo de computación de la repetición de Kernighan Lin es'+' '+str(Tiempo_computacion_KL))


import pickle
with open('comunidades_Kernighan_Lin.pickle', 'wb') as archivo:
    pickle.dump(comms_KL, archivo)
with open('Tiempo_Kernighan_Lin.pickle', 'wb') as archivo:
    pickle.dump(Tiempo_computacion_KL, archivo)
    



with open('comunidades_Kernighan_Lin.pickle', 'rb') as archivo:
    comunidades_Kernighan_Lin = pickle.load(archivo)


with open('comunidades_Kernighan_Lin.pickle', 'rb') as archivo:
    comunidades_Kernighan_Lin = pickle.load(archivo)

modularidad_KL=nx.community.modularity(G,comunidades_Kernighan_Lin)
print('La modularidad de la partición es'+ ' '+str(modularidad_KL))
numero_comunidades_KL=len(comunidades_Kernighan_Lin)
print('El número de comunidades de la partición es'+' '+str(numero_comunidades_KL))

with open('modularidad_K-L.pickle', 'wb') as archivo:
    pickle.dump(modularidad_KL, archivo)
with open('Número_comunidades_K-L.pickle', 'wb') as archivo:
    pickle.dump(numero_comunidades_KL, archivo)    
    
#Dibujo de las comunidades de Kernighan-Lin iterado
def comunidades_diccionario(comunidades):
    diccionario_comunidades=dict()
    for i in range(len(comunidades)):
        for nodo in comunidades[i]:
            diccionario_comunidades[nodo]=i
    return diccionario_comunidades


def dibujo_comunidades_KL(G, comunidades):
    pos = nx.random_layout(G,seed=5)
    cmap = cm.get_cmap('viridis', max(comunidades.values()) + 1)
    nx.draw_networkx_nodes(G, pos, comunidades.keys(), node_size=8,
    cmap=cmap, node_color=list(comunidades.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

dibujo_KL= dibujo_comunidades_KL(G, comunidades_diccionario(comunidades_Kernighan_Lin))




#Algoritmo GA-Net

def create_individuo(grafo):
    individuo=dict()
    for nodo in sorted(list(grafo.nodes())):
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

def reproduccion(padre1, padre2,grafo):
    hijo = dict()
    codific_binaria=random.choices([0,1],weights=[0.5,0.5],k=len(padre1))
    for i in range(len(padre1)):
        if codific_binaria[i]==1:
            hijo[sorted(list(grafo.nodes()))[i]]=padre1[sorted(list(grafo.nodes()))[i]]
        elif codific_binaria[i]==0:
            hijo[sorted(list(grafo.nodes()))[i]]=padre2[sorted(list(grafo.nodes()))[i]]
            
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
                hijo = reproduccion(progenitor1, progenitor2,grafo)
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
                    hijo = reproduccion(progenitor1, progenitor2,grafo)
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
inicio=time.time()
GA=genetic_algorithm(G,100, 10,0.1,0.8)
Tiempo_computacion_GA=time.time()-inicio
print('El timepo de computación de GA-Net es'+ ' '+str(Tiempo_computacion_GA))
print('La generación del mejor individuo es'+' '+str(GA[2]))
comms_GA=GA[1]
print('La partición encontrada por GA-Net es ')   
print(comms_GA)





import pickle
with open('comunidades_GA-Net.pickle', 'wb') as archivo:
    pickle.dump(comms_GA, archivo)
with open('Tiempo_GA-Net.pickle', 'wb') as archivo:
    pickle.dump(Tiempo_computacion_GA, archivo)
    




with open('comunidades_GA-Net.pickle', 'rb') as archivo:
    comunidades_GA = pickle.load(archivo)



with open('comunidades_GA-Net.pickle', 'rb') as archivo:
    comunidades_GA = pickle.load(archivo)

modularidad_GA=nx.community.modularity(G,comunidades_GA)
print('La modularidad de la partición es'+ ' '+str(modularidad_GA))
numero_comunidades_GA=len(comunidades_GA)
print('El número de comunidades de la partición del algoritmo genético es'+' '+str(numero_comunidades_GA))

with open('modularidad_GA-Net.pickle', 'wb') as archivo:
    pickle.dump(modularidad_GA, archivo)
with open('Número_comunidades_GA-Net.pickle', 'wb') as archivo:
    pickle.dump(numero_comunidades_GA, archivo)    
    
#Dibujo comunidades GA-Net
def comunidades_diccionario(comunidades):
    diccionario_comunidades=dict()
    for i in range(len(comunidades)):
        for nodo in comunidades[i]:
            diccionario_comunidades[nodo]=i
    return diccionario_comunidades


def dibujo_comunidades_GA(G, comunidades):
    pos = nx.random_layout(G,seed=5)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(comunidades.values()) + 1)
    nx.draw_networkx_nodes(G, pos, comunidades.keys(), node_size=8,
    cmap=cmap, node_color=list(comunidades.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

dibujo_KL= dibujo_comunidades_GA(G, comunidades_diccionario(comunidades_GA))


#Consideramos las comunidades con un tamaño igual osuperior a 15 nodos detectadas por cada algoritmo, lasrepresentamos g´raficamente junto con el nodo que mayor centralidad de grado tiene en cada comunidad

def comunidades_importantes(Grafo,partition):
    counter=0
    centers = {}
    communities = {}
    G_main_com =Grafo.copy()
    min_nb =15
    for com in set(partition.values()) :
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        if len(list_nodes) < min_nb:
            G_main_com.remove_nodes_from(list_nodes)
        else:
            H = G_main_com.subgraph(list_nodes)
            d_c = nx.degree_centrality(H)
            center = max(d_c, key=d_c.get)
            centers[center] = com
            communities[com] = center
            print('Community of ', center , '(ID ', com, ') - ', len(list_nodes), ' retweeters:')
            counter=counter+1
            print(list_nodes, '\n')
    print(counter)        
    return (G_main_com,centers,communities)





import matplotlib.pyplot as plt
import seaborn as sns
def dibujo_main_communities_with_centers(Grafo,centers,partition,communities):
    plt.figure(figsize=(20, 10))
    node_size = 50
    count = 0
    pos = nx.spring_layout(Grafo)
    colors = dict(zip(communities.keys(), sns.color_palette('hls', len(communities.keys()))))

    for com in communities.keys():
        count = count + 1
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com and nodes not in communities.values()]
        nx.draw_networkx_nodes(Grafo, pos, list_nodes, node_size = node_size, node_color = colors[com],alpha=0.6)
        nx.draw_networkx_nodes(Grafo, pos, list([communities[com]]), node_size = node_size*5, node_color = colors[com],alpha=0.6)
    nx.draw_networkx_edges(Grafo, pos, alpha=0.5)
    labels = {k: str(v) + ': ' + k for k,v in centers.items()}
    nx.draw_networkx_labels(Grafo, pos, labels,font_size=18,alpha=1)
    plt.axis('off')
    plt.show()
    
def comunidades_diccionario(comunidades):
    diccionario_comunidades=dict()
    for i in range(len(comunidades)):
        for nodo in comunidades[i]:
            diccionario_comunidades[nodo]=i
    return diccionario_comunidades




comunidades_importantes_louvain=comunidades_importantes(G,comunidades_diccionario(comunidades_louvain))
Grafo= comunidades_importantes_louvain[0]
centros=comunidades_importantes_louvain[1]
comunidades=comunidades_importantes_louvain[2]
particion=comunidades_diccionario(comunidades_louvain)
dibujo=dibujo_main_communities_with_centers(Grafo,centros,particion,comunidades)
import pickle
with open('comunidades_principales_louvain.pickle', 'wb') as archivo:
    pickle.dump(comunidades_importantes_louvain, archivo)



comunidades_importantes_louvain=comunidades_importantes(G,comunidades_diccionario(comunidades_louvain))
Grafo_parcial_Louvain= comunidades_importantes_louvain[0]
comunidades_principales=list()
for comunidad in comunidades_louvain:
    if len(list(comunidad))>=15:
        comunidades_principales.append(comunidad)
        
        
modularidad_comunidades_principales_Louvain=nx.community.modularity(Grafo_parcial_Louvain,comunidades_principales)  
print('La modularidad del subgrafo formado por las comunidades principales detectadas con el algoritmo de Louvain es:'+'\n '+str(modularidad_comunidades_principales_Louvain))




comunidades_importantes_GN=comunidades_importantes(G,comunidades_diccionario(comunidades_Girvan_Newman))
Grafo= comunidades_importantes_GN[0]
centros=comunidades_importantes_GN[1]
comunidades=comunidades_importantes_GN[2]
particion=comunidades_diccionario(comunidades_Girvan_Newman)
dibujo=dibujo_main_communities_with_centers(Grafo,centros,particion,comunidades)
import pickle
with open('comunidades_principales_GN.pickle', 'wb') as archivo:
    pickle.dump(comunidades_importantes_GN, archivo)



lista_nodos_comunidad=[]
for comunidad in comunidades_Girvan_Newman:
    lista_nodos_comunidad.append(len(comunidad))
print(lista_nodos_comunidad)    
    


comunidades_importantes_GN=comunidades_importantes(G,comunidades_diccionario(comunidades_Girvan_Newman))
Grafo_parcial_GN= comunidades_importantes_GN[0]
comunidades_principales=list()
for comunidad in comunidades_Girvan_Newman:
    if len(list(comunidad))>=15:
        comunidades_principales.append(comunidad)
        
        
modularidad_comunidades_principales_GN=nx.community.modularity(Grafo_parcial_GN,comunidades_principales)  
print('La modularidad del subgrafo formado por las comunidades principales detectadas con el algoritmo de G-N es:'+'\n '+str(modularidad_comunidades_principales_GN))




comunidades_importantes_KL=comunidades_importantes(G,comunidades_diccionario(comunidades_Kernighan_Lin))
Grafo= comunidades_importantes_KL[0]
centros=comunidades_importantes_KL[1]
comunidades=comunidades_importantes_KL[2]
particion=comunidades_diccionario(comunidades_Kernighan_Lin)
dibujo=dibujo_main_communities_with_centers(Grafo,centros,particion,comunidades)
import pickle
with open('comunidades_principales_KL.pickle', 'wb') as archivo:
    pickle.dump(comunidades_importantes_KL, archivo)



        
modularidad_comunidades_principales_KL=nx.community.modularity(Grafo_parcial_KL,comunidades_principales)  
print('La modularidad del subgrafo formado por las comunidades principales detectadas con el algoritmo de K-L es:'+'\n '+str(modularidad_comunidades_principales_KL))


        

comunidades_importantes_GA=comunidades_importantes(G,comunidades_diccionario(comunidades_GA))
Grafo= comunidades_importantes_GA[0]
centros=comunidades_importantes_GA[1]
comunidades=comunidades_importantes_GA[2]
particion=comunidades_diccionario(comunidades_GA)
dibujo=dibujo_main_communities_with_centers(Grafo,centros,particion,comunidades)
import pickle
with open('comunidades_principales_GA.pickle', 'wb') as archivo:
    pickle.dump(comunidades_importantes_GA, archivo)



comunidades_importantes_GA=comunidades_importantes(G,comunidades_diccionario(comunidades_GA))
Grafo_parcial_GA= comunidades_importantes_GA[0]
comunidades_principales=list()
for comunidad in comunidades_GA:
    if len(list(comunidad))>=15:
        comunidades_principales.append(comunidad)
        
        
modularidad_comunidades_principales_GA=nx.community.modularity(Grafo_parcial_GA,comunidades_principales)  
print('La modularidad del subgrafo formado por las comunidades principales detectadas con el algoritmo de GA-Net es:'+'\n '+str(modularidad_comunidades_principales_GA))

#Diccionarios y boxplot tamaños comunidades de cada algoritmo
import matplotlib.pyplot as plt
lista_nodos_comunidad_L=[]
for comunidad in comunidades_louvain:
    lista_nodos_comunidad_L.append(len(comunidad))
print(lista_nodos_comunidad_L)    
    
lista_nodos_comunidad_GN=[]
for comunidad in comunidades_Girvan_Newman:
    lista_nodos_comunidad_GN.append(len(comunidad))
print(lista_nodos_comunidad_GN)     

lista_nodos_comunidad_KL=[]
for comunidad in comunidades_Kernighan_Lin:
    lista_nodos_comunidad_KL.append(len(comunidad))
print(lista_nodos_comunidad_KL)      

lista_nodos_comunidad_GA=[]
for comunidad in comunidades_GA:
    lista_nodos_comunidad_GA.append(len(comunidad))
print(lista_nodos_comunidad_GA)
etiquetas=['Louvain', 'Girvan-Newman', 'Kernighan-Lin', 'GA-Net']
data=[lista_nodos_comunidad_L,lista_nodos_comunidad_GN,lista_nodos_comunidad_KL,lista_nodos_comunidad_GA]
bp=plt.boxplot(data, labels=etiquetas, showmeans=True, meanline=True)
bp
plt.title('Boxplot de numero nodos por comunidad')
plt.xlabel('Algoritmos')
plt.ylabel('Nodos por comunidad')


mediana = bp['medians'][0]

media = bp['means'][0]

labels = ['Medianas', 'Medias' ]

plt.legend([mediana , media], labels, loc='upper right')

plt.show()


def conteo_comunidades(lista):
    cantidades=dict()
    for n_nodos in lista:
        if n_nodos not in cantidades:
            cantidades[n_nodos]=1
        else:
            cantidades[n_nodos]+=1
    return cantidades        
        
print('Diccionario cantidad de comunidades por tamaño Louvain')
print(conteo_comunidades(lista_nodos_comunidad_L))
print('Diccionario cantidad de comunidades por tamaño Girvan-Newman')
print(conteo_comunidades(lista_nodos_comunidad_GN))
print('Diccionario cantidad de comunidades por tamaño Kernighan-Lin iterado')
print(conteo_comunidades(lista_nodos_comunidad_KL))
print('Diccionario cantidad de comunidades por tamaño GA-Net')
print(conteo_comunidades(lista_nodos_comunidad_GA))


