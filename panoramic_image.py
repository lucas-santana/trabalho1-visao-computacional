

import cv2 as cv
import numpy as np
import math

def getPanoramicImage(H, imgInput1, imgInput2):
    """
        mapeamento inverso
            Preciso aplicar a homografia na imagem resultante e verificar se cai na imagem1 ou na imagem 2
    """
    
    inverse_homografy = np.linalg.inv(H)
    
    # Largura da nova imagem é a junção das 2
    new_width = imgInput1.shape[1] + imgInput2.shape[1]
    
    # Pega a maior altura entre as 2 imagens
    new_height = max(imgInput1.shape[0], imgInput2.shape[0])
    
    img_stitch = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    img_stitch[0:imgInput1.shape[0], 0:imgInput1.shape[1]] = imgInput1
    
    # calculo do mapeamento inverso: da imagem 2 para a imagem 1
    # inv_img2 = [] # Mapeamento 
    # for linha_i in range(imgInput2.shape[0]):
    #     for coluna_j in range(imgInput2.shape[1]):
    #             inverse_mapped_point = np.dot(inverse_homografy, np.array([linha_i, coluna_j, 1]))
    #             inverse_mapped_point = (1 / inverse_mapped_point[-1]) * inverse_mapped_point
    #             print("Linha: ", linha_i)
    #             print("Coluna: ", coluna_j)
    #             print("mapeamento inverso: ", inverse_mapped_point)
    #             exit(0)
    
    print("Dimensoes imagem resultante", img_stitch.shape)
    for linha_i in range(new_height):
        for coluna_j in range(new_width):
            if coluna_j > imgInput1.shape[1]:#ponto da imagem da direita
                
                # aplicar homografia inversa para fazer mapeamento inverso
                inverse_mapped_point = np.dot(inverse_homografy, np.array([linha_i, coluna_j, 1]))
                inverse_mapped_point = (1 / inverse_mapped_point[-1]) * inverse_mapped_point
                
                # aplicar homografia para pegar o ponto correspondente na imagem 2
                direct_mapped_point = np.dot(H, np.array([inverse_mapped_point[0], inverse_mapped_point[1], 1]))
                direct_mapped_point = (1 / direct_mapped_point[-1]) * direct_mapped_point
                
                if(check_mapped_point(direct_mapped_point, imgInput2)):
                    print("Linha: ", linha_i)
                    print("Coluna: ", coluna_j)
                    print("mapeamento inverso: ", inverse_mapped_point)
                    print("mapeamento direto: ", direct_mapped_point)
                    exit(0)
                    
                    
                    
                
                
                # pegar o pixel na imagem 2 e salvar na linha_i/coluna_j da imagem resultante
                
                
                
    
    # Aplicar H inversa nos extremos da imagem da direita e verificar altura/largura máxima
    for linha_i in range(new_height):
        for coluna_j in range(new_width):
            
            inverse_mapped_point = np.dot(inverse_homografy, np.array([linha_i, coluna_j, 1]))
            inverse_mapped_point = (1 / inverse_mapped_point[-1]) * inverse_mapped_point
            # inverse_mapped_point = (np.round(inverse_mapped_point)).astype(int) # Arredondar e converter para int
            
            # Verifica se o ponto na imagem resultante está na esquerda da imagem1
            if coluna_j < imgInput1.shape[1]:
                
                # Verifica se foi mapeado dentro de um local da imagem 1
                if(check_mapped_point(inverse_mapped_point, imgInput1)):
                    
                    # Faz a interpolação com os pixels na imagem da esquerda
                    rgb = bilinear_interpolation(imgInput1, inverse_mapped_point)

                else:
                    # Está na imagem da esquerda mas o ponto mapeado está fora da imagem da esquerda
                    # pega o pixel da imagem da esquerda sem interpolar
                    rgb = imgInput1[linha_i][coluna_j]
                    
            else: # ponto na imagem resultante está na imagem da direita
                
                # verifica se o ponto mapeado está na imagem da esquerda
                if(check_mapped_point(inverse_mapped_point, imgInput1)):
                    # ponto na imagem resultante está na imagem da direita e o ponto mapeado está na imagem da esquerda
                    
                    # Faz a interpolação com os pixels na imagem da direita
                    rgb = bilinear_interpolation(imgInput2, inverse_mapped_point)
    
    # img_stitch[0:imgInput1.shape[0], imgInput1.shape[1] : new_width] = imgInput2
    
    return img_stitch

def getPanoramicImageWithOpenCV(H, imgInput1, imgInput2):
    
    # Largura da nova imagem é a junção das 2
    width = imgInput1.shape[1] + imgInput2.shape[1]
    
    # Pega a maior altura entre as 2 imagens
    height = max(imgInput1.shape[0], imgInput2.shape[0])
    
    #
    img_stitch = cv.warpPerspective(imgInput2, np.linalg.inv(H), (width, height))
    
    # Substitui na imagem final o lado esquerdo pela imagem 1
    img_stitch[0:imgInput1.shape[0], 0:imgInput1.shape[1]] = imgInput1
    
    return img_stitch

def bilinear_interpolation(imgInput1, inverse_mapped_point):
    
    return 0

def check_mapped_point(inverse_mapped_point, imgInput):
    """Verifica se o o ponto mapeado está dentro de imgInput

    Args:
        inverse_mapped_point (_type_): _description_
        imgInput (_type_): _description_

    Returns:
        _type_: _description_
    """
    if(inverse_mapped_point[0] > 0 and inverse_mapped_point[0] < imgInput.shape[0] # altura/linhas
               and inverse_mapped_point[1] > 0 and inverse_mapped_point[1] < imgInput.shape[1]): # largura/colunas
        return True
    return False