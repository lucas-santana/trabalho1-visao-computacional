import numpy as np
import math

def ransac_calc(all_query_points, all_train_points):
    """
        Recebe como parametro os pontos de correspondencia encontrados
    """

    print("Cálculo da Homografia por RANSAC\n")
    
    
    best_homography, best_inliers = get_best_homografy(all_query_points, all_train_points)
    
    if len(best_inliers) > 0:
        best_query_points = best_inliers[:,:2]
        best_train_points = best_inliers[:,2:]
        
        # Calcular homografia com todos inliers
        H = get_homografy(best_query_points, best_train_points)
        
        return H
    else:
        print("Homografia com boa quantidade de inliers não encontrada...")
        return None
    
    """
    # Debug para saber se a homografia está mapeando os pontos corretamente
    np.random.seed(29) # Semente para sempre gerar os mesmos valores a cada execução
    rand_idx = np.random.randint(len(all_query_points), size = 4)
        
    query_points = []
    train_points = []
    for i in range(4):
        query_points.append(all_query_points[rand_idx[i]])
        train_points.append(all_train_points[rand_idx[i]])
    query_points = np.array((query_points))
    train_points = np.array((train_points))
    
    H = get_homoGrafy(query_points, train_points)
    
    for i in range(4):
        print(f"query[{i}]: ", query_points[i])
        print(f"train[{i}]: ", train_points[i])
        print("Erro: ", geometricDistance(query_points[i], train_points[i], H ))
        print("\n")
    
    return H
    """

# Verificação para não gerar os mesmos índices
def checkIndices(indices, rand_idx):
    for sublist in indices:
        rand_idx.sort()
        sublist.sort()
        if ((rand_idx == sublist).all()):
            return False
    return True
    
def get_best_homografy(all_query_points, all_train_points):
    print("Encontrar melhor homografia")
    """
        Obtém a melhor Homografia H pelo método do RANSAC
    """
    best_inliers = []
    best_homography = None
    sample_count = 0
    E = 1.0
    N = math.inf
    s = 4
    p = 0.99
    menor = math.inf
    inlier_threshold = np.sqrt(6)
    
    #np.random.seed(29) Semente para sempre gerar os mesmos valores a cada execução
    
    indices = []
    while(N > sample_count):
        """
            Gera 4 indices aleatórios e pega os pontos associados a esses indices
        """
        rand_idx = np.random.randint(len(all_query_points), size = s)
        
        # while(checkIndices(indices, rand_idx) == False):
        #     rand_idx = np.random.randint(len(all_query_points), size = s)
        #     print("Indices já foram analisados")
        # indices.append(rand_idx)
        
        query_points = []
        train_points = []
        for i in range(s):
            query_points.append(all_query_points[rand_idx[i]])
            train_points.append(all_train_points[rand_idx[i]])

        query_points = np.array((query_points))
        train_points = np.array((train_points))

        H = get_homografy(query_points, train_points)
        
        
        inliers = []
        for q, t in zip(all_query_points, all_train_points):
            d = geometricDistance(q, t, H)
            if(d <= inlier_threshold):
                c = np.hstack((q, t))
                inliers.append(c)
            else:
                if menor > d:
                    menor = d
            
        inliers = np.array(inliers)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = H
            print(f"N: {N} \t\t sample_count: {sample_count} \t\t inliers:{len(inliers)} \t\t best_inliers: {len(best_inliers)} \t\t Matches: {len(all_query_points)}")

        if len(best_inliers) == (len(all_query_points)):
            print("Encontrou todos pontos!")
            break
    
        """
            Queremos a probabilidade p de que pelo menos uma das amostras seja formada por s=4 pontos e nenhum desses 4 seja um outlier
            p =     probabilidade de só ter inlier
            1-p =   prob de ter pelo menos 1 outlier em cada amostra
            E = probabilidade de ser outlier
        """
        Ei = 1 - (len(inliers)/len(all_query_points)) # proporção de outliers
        if Ei < E:
            E = Ei
            nume = np.log(1 - p)
            den = (1-E)**s
            den = 1 - den
            den = np.log(den)
            N = int(nume/den)
            
        
        sample_count = sample_count + 1
        if sample_count == 1 or sample_count % 1000 == 0:    
            print(f"N: {N} \t\t sample_count: {sample_count} \t\t inliers:{len(inliers)} \t\t best_inliers: {len(best_inliers)} \t\t Matches: {len(all_query_points)}")

    print(f"\nNum matches: {len(all_query_points)}")
    print(f"Num inliers: {len(best_inliers)}")
    print(f"N final: {N}")
    print("Menor d: ", menor)

    """
        Rodar a solução usando os best_inliers
        No caso da homografia, estimar a melhor solução usando esses inliers, ou seja,
        achar a homografia com os best_inliers em vez de somente 4 pontos
    """
    return best_homography, best_inliers


def get_homografy(query_points, train_points):
    """
        Calcula homografia dos n pontos de correspondencia passados como parametro
    """
    
    if(len(query_points) < 4 or len(train_points) < 4):
        print("Informe no minimo 4 pontos...")
    # TODO: Testar retornando o T aqui pra nao precisar calcular novamente
    # Normalization of query and train points
    normalized_query, T1 = normalize(query_points)
    normalized_train, T2 = normalize(train_points)
    
    # normalized_query = query_points
    # normalized_train = train_points

    # print("Distancia média query: ", np.mean( np.linalg.norm(normalized_query, axis=1)))
    # print("Distancia média train: ", np.mean( np.linalg.norm(normalized_train, axis=1)))

    correspondences = []
    for i in range(len(query_points)):
        c = np.hstack((normalized_query[i], normalized_train[i]))
        correspondences.append(c)
    
    correspondences = np.array(correspondences)

    # Montar a matriz A 2nx9 onde n é o numero de correspondencias
    A = np.empty((0, 9))
    for corr in correspondences:
        p_q = corr[:2] # pontos  na imagem query
        p_t = corr[2:] # pontos na imagem de treinamento

        eq_1 = np.array([p_q[0], p_q[1], 1, 0, 0, 0, -p_t[0]*p_q[0], -p_t[0]*p_q[1], -p_t[0]])
        eq_2 = np.array([0, 0, 0, p_q[0], p_q[1], 1, -p_t[1]*p_q[0], -p_t[1]*p_q[1], -p_t[1]])
        A = np.vstack((A, eq_1, eq_2))

    # Obter o SVD de A
    U, S, V = np.linalg.svd(A)
    
    """
    The unit singular vector corresponding to the smallest singular value is the solution h. 
    Specifically, if A = UDV^T with D diagonal with positive diagonal entries, arranged in descending order down the diagonal, 
    then h is the last column of V
    Então pegamos apenas a ultima coluna de V e mudamos para um formato 3x3
    """
    H = np.reshape(V[8], (3, 3))
    H = (1/H[2, 2]) * H
    
    """
        Alternativa para calcular h, considerando que h  ́e definido a menos de uma escala,
    
    # Calcula a matriz A^T A
    ATA = np.dot(A.T, A)

    # Calcula autovalores e autovetores de ATA
    autovalores, autovetores = np.linalg.eig(ATA)

    # Encontra o índice do menor autovalor
    indice_menor_autovalor = np.argmin(autovalores)

    # Obtém o autovetor correspondente ao menor autovalor
    autovetor_menor_autovalor = autovetores[:, indice_menor_autovalor]
    
    H = np.reshape(autovetor_menor_autovalor, (3, 3))
    """
    
    # Denormalization
    H = denormalize(T1, T2, H)
    

    return H

def algebraicDistance(A, H):
    """
    Calcula a norma do vetor de erro dado por E = A*H, que é a distância algebrica
    
    Referencia: Seção 4.2.1 do HARTLEY; ZISSERMAN (2004)
    
    Parameters
    ------
    A: matriz no formato nx9
        Matriz de equações das correspondências
    
    H: vetor homografia no formato 3x3
    
    Return
    --------
    n : ndarray
        Norma do residuo A*H
    
    """
    erro = np.dot(A, H.reshape(9,1))
    
    return np.linalg.norm(erro)

def geometricDistance(p_query, p_train, H):
    """
        Aplico H no ponto da imagem query. Isso deveria me dar o ponto pt da imagem train
        Então calculo a distancia entre o calculado e o real
        Esta é a distância euclidiana  na segunda imagem entre o ponto medido pt e o ponto 
        pt_estimate no qual o ponto correspondente pq é mapeado a partir da primeira imagem.
        
        Referencia: Seção 4.2.2 do HARTLEY; ZISSERMAN (2004)
    """
    pq = np.array([p_query[0], p_query[1], 1])
    pt = np.array([p_train[0], p_train[1], 1])
    
    pt_estimated = np.dot(H, pq)
    pt_estimated = (1 / pt_estimated[2]) * pt_estimated
    
    # print("PT: ", pt)
    # print("ESTIMADO: ", pt_estimated)
    error = np.linalg.norm(pt - pt_estimated)
    
    return error


"""
    isotropic scaling - pag 107 4.4.4 Normalizing transformations do HARTLEY; ZISSERMAN (2004),
    
"""
def getTransformationT(points):
    """
        The points are translated so that their centroid is at the origin.
    """
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    
    points = points - [mean_x, mean_y]

    # The points are then scaled so that the average distance from the origin is equal to √2.
    distances = np.linalg.norm(points, axis=1) # distancia de cada ponto até a origem
    
    avg_distance = np.average(distances)
    
    scale_factor = avg_distance/np.sqrt(2)
    
    T = np.array([
                    [scale_factor, 0, -mean_x],
                    [0, scale_factor, -mean_y],
                    [0, 0, 1]
                ])
    
    return T

"""
    Compute a similarity transformation T, consisting of a translation
    and scaling, that takes points xi to a new set of points x2i such that the centroid of the
    points x2i is the coordinate origin (0, 0)T, and their average distance from the origin is
    √2.
"""

def normalizeOld(points):
    T = getTransformationT(points) # matriz 3x3
    
    in1 = points # matriz com os 4 pontos da correspondência
    in2  = np.ones((len(points), 1)) # cria vetor de 1 do mesmo tamanho do numero de pontos
    homogeneous_points = np.hstack((in1, in2)) # adiciona uma coluna de 1 na matriz de pontos - coordenadas homogênea
    
    # Aplica a Transformação em cada ponto
    for i in range(4):
        homogeneous_points[i] = np.dot(T, homogeneous_points[i])

    homogeneous_points = homogeneous_points[:, :2] # remover coluna homogenea adicionada

    return homogeneous_points, T

def normalize(points):
    mean_x = np.mean(points[:, 0])
    mean_y = np.mean(points[:, 1])
    mean = [mean_x, mean_y]
    
    center_points = points - mean
    
    distance = np.linalg.norm(center_points, axis=1)
    
    avg_distance = np.sum(distance)/len(points)
    
    scale_factor = avg_distance / np.sqrt(2)

    T = np.array([[scale_factor, 0, mean[0]],
                  [0, scale_factor, mean[1]],
                  [0, 0, 1]])
    T = np.linalg.inv(T)
    
    n1 = points
    n2 = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((n1, n2))
    
    points = np.dot(T, homogeneous_points.T)
    points = points[0:2].T
    
    return points, T

def denormalize(T1, T2, H):
    H = np.dot(np.dot(np.linalg.inv(T2), H), T1)
    H = H/H[-1, -1]
    return H