#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Algoritmo Louvain
import networkx as nx
import community
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def fase_uno(grafo, particion_general):
    partition={node:node for node in grafo.nodes}   
    nodes = list(grafo.nodes)
    random.shuffle(nodes)
    improved=True
    q=0
    while(improved==True):
        p=0
        improved=False
        for node in nodes:
            comunidad_inicial = partition[node]
            best_community = comunidad_inicial
            max_incremento_Q=0.0
            vecinos=list(grafo.neighbors(node))
            for vecino in vecinos:
                 if partition[vecino] != comunidad_inicial:
                    nueva_particion_posible=partition.copy()
                    nueva_particion_posible[node]=partition[vecino]
                    modularidad_actual=nx.community.modularity(grafo, convertir_a_lista_comunidades(partition), weight='weight', resolution=1)
                    modularidad_nueva=nx.community.modularity(grafo,  convertir_a_lista_comunidades(nueva_particion_posible), weight='weight', resolution=1)
                    incremento= modularidad_nueva-modularidad_actual
                    print(incremento)
                    if incremento > max_incremento_Q:
                        max_incremento_Q = incremento
                        best_community = partition[vecino]

            # Realizar el movimiento que maximice la modularidad
            print(max_incremento_Q)
            if max_incremento_Q > 0.0:
                partition[node] = best_community
                improved=True
            p=p+1
        print('nodos vistos en esta ronda'+ str(p))
        q=q+1
        print('acabada ronda'+ str(q))
    particion_general=actualizacion_particion(partition, particion_general)
    
    
    return(partition,particion_general)

    
    
def aumento_Q(grafo,particion,vecino,nodo):
    comunidad=particion[vecino]
    sum_in=suma_pesos_in_C(grafo,particion,comunidad)
    sum_tot=suma_pesos_total(grafo,particion,comunidad)
    ki_in=suma_pesos_aristas_nodo_comunidad(grafo,particion,comunidad,nodo)
    ki=suma_pesos_aristas_nodo(grafo,nodo)
    m=suma_pesos_grafo(grafo)
    primer_termino=((sum_in+(2*ki_in))/(2*m))-(((sum_tot+ki)/(2*m))**2)
    segundo_termino=(sum_in/(2*m))-((sum_tot/(2*m))**2)-((ki/(2*m))**2)
    delta_Q= primer_termino - segundo_termino
    return delta_Q

def suma_pesos_grafo(grafo):
    suma=0
    for arista in grafo.edges():
        suma=suma+peso_arista(grafo,arista)
    return suma

def peso_arista(grafo,arista):
    peso= grafo.edges[arista[0], arista[1]]['weight']
    return peso


def aristas_en_comunidad(grafo,particion,comunidad):
    nodos_comunidad=[]
    for nodo in list(grafo.nodes):
        if particion[nodo]==comunidad:
            nodos_comunidad.append(nodo)
            
    subgrafo=grafo.subgraph(nodos_comunidad)
    aristas=subgrafo.edges()
    return(aristas)        
  

def suma_pesos_in_C(grafo,particion,comunidad):
    suma=0
    edges_en_comunidad=aristas_en_comunidad(grafo,particion,comunidad)
    for arista in  edges_en_comunidad:
        suma=suma+peso_arista(grafo,arista)
        
    return suma

def aristas_incidentes_comunidad(grafo,particion,comunidad):
    aristas=[]
    for arista in grafo.edges():
        if particion[arista[0]]==comunidad or particion[arista[1]]==comunidad :
            aristas.append(arista)
    return (aristas)


def suma_pesos_total(grafo,particion,comunidad):
    suma=0
    aristas=aristas_incidentes_comunidad(grafo,particion,comunidad)
    for arista in aristas:
        suma= suma + peso_arista(grafo,arista)
    return suma

def aristas_nodo_comunidad(grafo,particion,comunidad,nodo):
    aristas=[]
    for arista in grafo.edges():
        if arista[0]==nodo and particion[arista[1]]==comunidad :
            aristas.append(arista)
        elif arista[1]==nodo and particion[arista[0]]==comunidad:
            aristas.append(arista)
    return aristas      


def suma_pesos_aristas_nodo_comunidad(grafo,particion,comunidad,nodo):
    suma=0
    aristas= aristas_nodo_comunidad(grafo,particion,comunidad,nodo)
    for arista in aristas:
        suma=suma+peso_arista(grafo,arista)
    return suma

def aristas_nodo(grafo,nodo):
    aristas=[]
    for arista in grafo.edges():
        if arista[0]==nodo or arista[1]==nodo:
            aristas.append(arista)
    return aristas            
def suma_pesos_aristas_nodo(grafo,nodo):
    suma=0
    aristas=aristas_nodo(grafo,nodo)
    for arista in aristas:
        suma=suma+peso_arista(grafo,arista)
    return suma  



def fase_dos(particion,grafo,particion_g):
    G=nx.Graph()
    nodos=[]
    edges=[]
    for community in set(particion.values()):
        nodos.append(community)
    for community1 in set(particion.values()):
        for community2 in set(particion.values()):
            if aristas_entre_comunidades(community1, community2,particion,grafo)!=[]:
                if (community1,community2) not in edges and (community2,community1) not in edges:
                    edges.append((community1,community2))
                
    G.add_nodes_from(nodos)
    G.add_edges_from(edges)
    
    for arista in G.edges():
        G.edges[arista[0], arista[1]]['weight']=suma_pesos_aristas_entre_comunidades(arista[0],arista[1],grafo,particion)
    
    return(G)

    if grafo!=G:
        new_partition,particion_final=fase_uno(G,particion_g)
        fase_2=fase_dos(new_partition,G,particion_final)
        
    return particion_final   

def aristas_entre_comunidades(comunidad1, comunidad2,particion,grafo):
    aristas=[]
    for arista in grafo.edges():
        if particion[arista[0]]== comunidad1 and particion[arista[1]]==comunidad2:
            aristas.append(arista)
        elif particion[arista[1]]== comunidad1 and particion[arista[0]]==comunidad2:
            aristas.append(arista)    
    return aristas         
def suma_pesos_aristas_entre_comunidades(comunidad1,comunidad2,grafo,particion):
    suma=0
    for arista in aristas_entre_comunidades(comunidad1, comunidad2,particion,grafo):
        if arista[0]==arista[1]:
            suma=suma+2*peso_arista(grafo,arista)
        else:
             suma=suma+peso_arista(grafo,arista)
    return suma    

def actualizacion_particion(particion1, particion_g):
    for nodo in particion_g.keys():
        particion_g[nodo]=particion1[particion_g[nodo]]
    return particion_g


def algoritmo_louvain(grafo_inicial, particion_inicial):
    i=1
    antes_f2 = grafo_inicial
    first_f1 = fase_uno(grafo_inicial, particion_inicial)
    dibujo_comunidades(grafo_inicial, first_f1[1])
    first_f2 = fase_dos(first_f1[0], grafo_inicial, first_f1[1])
    diccionario_comunidades={nodo:nodo for nodo in list(first_f2.nodes())}
    dibujo_comunidades(first_f2, diccionario_comunidades)
    while nx.is_isomorphic(antes_f2, first_f2)==False:
        i=i+1
        antes_f2 = first_f2
        first_f1 = fase_uno(first_f2, first_f1[1])
        print('Dibujo fase uno general')
        dibujo_comunidades(grafo_inicial, first_f1[1])
        first_f2 = fase_dos(first_f1[0], first_f2, first_f1[1])
        diccionario_comunidades={nodo:nodo for nodo in list(first_f2.nodes())}
        print('Dibujo fase dos')
        dibujo_comunidades(first_f2, diccionario_comunidades)
    return first_f1[1], i

    


def convertir_a_lista_comunidades(diccionario_comunidades):
    comunidades = {}  
    for nodo, comunidad in diccionario_comunidades.items():
        if comunidad not in comunidades:
            comunidades[comunidad] = set()  
        comunidades[comunidad].add(nodo)  
    
    lista_comunidades = [set(nodos) for nodos in comunidades.values()]
    return lista_comunidades

particion_in=dict()
for node in graph.nodes():
    particion_in[node]=node
    
    




#Algoritmo Kernighan-Lin
from IPython.core.debugger import set_trace as breakpoint
    
def KerninghanLin(grafo):
    #particion= particion_aleatoria(grafo)
    particion=particion_aleatoria(grafo)
    dibujo_particion_inicial=dibujo_comunidades(grafo, particion)
    particiones=[]
    
    p = 0 # pass
    total_gain = 0
    stop=0
    while stop==0:
        stop=1
        A_ = []
        B_ = []
        
        for node in grafo.nodes():
            if particion[node] == 0:
                A_.append(node)
            elif particion[node]== 1:
                B_.append(node)
        
        valores_D = {node: valor_D(A_,B_,grafo,node) for node in grafo.nodes()}
        print(valores_D)
        ganancias = [] # [ ([a, b], gain), ... ]
        
        for _ in range(int(len(grafo.nodes())/2)): 
            
        # choose a pair that maximizes gain 
            ganancia_maxima = -1 * float("inf") # -infinity
            par_nodos = []
            
            for nodo_a in A_:
                for nodo_b in B_:
                    c_ab = valor_Cij(nodo_a,nodo_b,grafo)
                    ganancia = valores_D[nodo_a] + valores_D[nodo_b] - (2 * c_ab)                   
                    if ganancia > ganancia_maxima:
                        ganancia_maxima = ganancia
                        par_nodos = [nodo_a,nodo_b] 
            
            A_.remove(par_nodos[0])
            B_.remove(par_nodos[1])
            ganancias.append([par_nodos,ganancia_maxima])
            print(ganancias)
        
            for x in A_:
                c_xa = valor_Cij(par_nodos[0],x,grafo)
                c_xb = valor_Cij(par_nodos[1],x,grafo)
                valores_D[x] += 2 * (c_xa) - 2 * (c_xb)
            
            for y in B_:
                c_ya = valor_Cij(par_nodos[0],y,grafo)
                c_yb = valor_Cij(par_nodos[1],y,grafo)
                valores_D[y] += 2 * (c_yb) - 2 * (c_ya)
        
           
        g_max = -1 * float("inf")
        jmax = 0
        for j in range(1,len(ganancias)+1):
            suma_ganancias = 0
            for k in range(j): 
                suma_ganancias += ganancias[k][1]
            
            if suma_ganancias > g_max:
                g_max = suma_ganancias
                jmax = j
        print(g_max)
        if g_max > 0:
            for i in range(jmax):
                for node in grafo.nodes():
                    if node==ganancias[i][0][0]:
                        particion[node]=1
                    elif node==ganancias[i][0][1]:
                        particion[node]=0    
 
            p += 1
            total_gain += g_max
            print ("Pass: " + str(p) + "\t\t\tGain: " + str(g_max))
            dibujo=dibujo_comunidades(grafo, particion)
            stop=0
            
        else: 
            break
        
    return particion
        
        


def particion_aleatoria(grafo):
    length=int(len(grafo.nodes())/2)
    random_A = random.sample(list(grafo.nodes()),length)
    particion={node: 0 if node in random_A else 1 for node in grafo.nodes()}
    return(particion)

def coste_particion(grafo,particion):
    coste = 0
    for edge in grafo.edges():
        if particion[edge[0]]!=particion[edge[1]]:
            coste += 1   
    return coste

def valor_D(nodos_A,nodos_B,grafo,nodo):
    D=0
    grupo_nodes=dict()
    for node in grafo.nodes():
        if node in nodos_A:
            grupo_nodes[node]="A"
        elif node in nodos_B:
             grupo_nodes[node]="B"
    for arista in grafo.edges():
        if arista[0]==nodo:
            if grupo_nodes[arista[1]]==grupo_nodes[nodo]:
                D=D-1
            elif  grupo_nodes[arista[1]]!=grupo_nodes[nodo]:
                D=D+1
        elif arista[1]==nodo:
            if grupo_nodes[arista[0]]==grupo_nodes[nodo]:
                D=D-1
            elif  grupo_nodes[arista[0]]!=grupo_nodes[nodo]:
                D=D+1
    return D

def valor_Cij(nodo1,nodo2,grafo):
    valor=0
    for arista in grafo.edges():
        if arista[0]==nodo1 and arista[1]==nodo2:
            valor=valor+1
        elif arista[0]==nodo2 and arista[1]==nodo1:
            valor=valor+1     
    
    return valor




#GA-Net

def dirac(nodo1,nodo2,comunidades):
    comunidad_nodo1= comunidades[nodo1]
    comunidad_nodo2=comunidades[nodo2]   
    resultado=0
    if comunidad_nodo1==comunidad_nodo2:
        resultado=1
    else:
        resultado=0
        
    return(resultado)  

def matriz_adyacencia(grafo):
    matriz=np.zeros((int(len(grafo.nodes())),int(len(grafo.nodes()))))
    aristas=list(grafo.edges())
    for arista in aristas:
        matriz[arista[0],arista[1]]=1
        matriz[arista[1],arista[0]]=1
    return matriz   


def create_individuo(grafo):
    individuo=dict()
    for nodo in grafo.nodes():
        gen=random.choice(list(grafo.neighbors(nodo)))
        individuo[nodo]=gen
    return individuo    


def primera_generacion(num_individuos,grafo):
    generacion_1 = []
    while len(generacion_1)< num_individuos :
        individuo = create_individuo(grafo)
        if individuo not in generacion_1:
            generacion_1.append(individuo)
    return generacion_1

def seleccionar_padres(poblacion, evaluaciones):
    total=sum(evaluaciones)
    probabilidades = [evaluacion / total for evaluacion in evaluaciones]
    padres_seleccionados = random.choices(poblacion, weights=probabilidades, k=2)

    return padres_seleccionados

def seleccionar_individuos(poblacion, evaluaciones,n):
    total=sum(evaluaciones)
    probabilidades = [evaluacion / total for evaluacion in evaluaciones]
    individuos_seleccionados = random.choices(poblacion, weights=probabilidades, k=n)
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
    for i in range(len(individuo)):
        n=random.uniform(0,1)
        gen = individuo[i]
        if n <= probabilidad_mutacion: 
            vecinos= grafo.neighbors(i)
            individuo_mutado[i] = random.choice(list(vecinos))
    return individuo_mutado


def genetic_algorithm(grafo,num_individuos, num_generaciones,probabilidad_mutacion,r,crossover_rate):
    A=matriz_adyacencia(grafo)
    poblacion= primera_generacion(num_individuos,grafo)
    print('La primera generación es')
    print(poblacion)
    best_fitness = float('-inf')
    mejor_individuo = None
    mejores_individuos_generaciones=[]
    c=1
    for generacion in range(num_generaciones-1):
        individuos_evaluados = dict()
        for individuo in poblacion:
            fitness = community_score(individuo_comunidades(individuo),A,r)
            individuos_evaluados[fitness]=individuo
            if fitness >= best_fitness:
                best_fitness = fitness
                mejor_individuo = individuo
                generacion_mejor_individuo= generacion +1
                
        siguiente_generacion = []
        while len(siguiente_generacion) < num_individuos:
            progenitor1, progenitor2 = seleccionar_padres(list(individuos_evaluados.values()), list(individuos_evaluados.keys()))
            if random.uniform(0,1) < crossover_rate:
                hijo = reproduccion(progenitor1, progenitor2)
                mutacion_hijo = mutacion(hijo, probabilidad_mutacion,grafo)
                siguiente_generacion.append(mutacion_hijo)
            else:
                siguiente_generacion.append(0)
        
        count=0
        for elemento in  siguiente_generacion:
            if elemento==0:
                count=count+1
                siguiente_generacion.remove(elemento)
        
        individuos_no_mutados= seleccionar_individuos(list(individuos_evaluados.values()), list(individuos_evaluados.keys()),count)
        siguiente_generacion.extend(list(individuos_no_mutados))
        
        poblacion = siguiente_generacion
        c=c+1
        print('generación'+str(c))
        print(siguiente_generacion)
    
    return mejor_individuo,generacion_mejor_individuo,best_fitness 

def individuo_comunidades(individuo):
    comunidades=[]
    copia_individuo=individuo.copy()
    
    while copia_individuo!={}:
        nodo=list(copia_individuo.keys())[0]
        comunidad=[]
        comunidad_anterior=[]
        comunidad.append(nodo)
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

def M_S(comunidad,A,r):
    resultado=0
    suma_aiJ=0
    for i in comunidad:
        suma_aiJ=suma_aiJ +((a_iJ(comunidad,i,A))**r)
        
    resultado=resultado +(suma_aiJ/len(comunidad)) 
    return resultado

def a_iJ(comunidad,nodo_i,A):
    result=0
    suma_aij=0
    for j in comunidad:
        suma_aij=suma_aij + A[nodo_i,j]
    
    result=result+(suma_aij/len(comunidad))
    return result

def volumen_S(comunidad,A):
    volumen=0
    for nodo_k in comunidad:
        for nodo_l in comunidad:
            volumen=volumen+A[nodo_k,nodo_l]
    return volumen


def community_score(comunidades,A,r):
    score=0
    for k in range(len(comunidades)):
        score=score+(volumen_S(comunidades[k],A)*M_S(comunidades[k],A,r))
    return score



def dibujo_grafo(grafo):
    pos = nx.circular_layout(grafo)
    #pos = nx.spring_layout(grafo)
    nx.draw_networkx(grafo, pos,node_size = 90,node_color='yellowgreen', alpha = 0.9,with_labels=True,font_size=9)
    plt.show()
dibujo= dibujo_grafo(graph)

def dibujo_comunidades(grafo, diccionario_comunidades):
    pos = nx.circular_layout(grafo)
    #pos = nx.spring_layout(grafo)
    # color the nodes according to their partition
    cmap = cm.get_cmap('Set3', max(diccionario_comunidades.values()) + 1)
    nx.draw_networkx(grafo, pos, diccionario_comunidades.keys(), node_size=75, cmap=cmap,
    node_color=list(diccionario_comunidades.values()),with_labels=True)
    plt.show()


#Algoritmo Girvan-Newman
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
        print(max_edge_betweennes(edge_betweenness))
        grafo_copia.remove_edge(edge_remove[0],edge_remove[1])
        componentes=list(nx.connected_components(grafo_copia))
        print(componentes)
        if len(componentes) > len (particion_actual):
            particion_actual= componentes
            particiones.append(componentes)
            
    return particiones

