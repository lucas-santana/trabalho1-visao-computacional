
# Para executar:
# python main.py img/leao.jpg img/leao.jpg orb
import sys
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from features import matching
from homografy import ransac_calc
from panoramic_image import getPanoramicImage, getPanoramicImageWithOpenCV

def create_output_dir(name):
    if not os.path.exists("output/"+name):
        os.makedirs("output/"+name)

def main():
    """
        Execução do programa com ORB
            python main.py T2 orb 
        Execução do programa com SIFT
            python main.py T2 sift
    """
    test_image_folder = sys.argv[1] # diretório onde estão as imagens de teste (T1, T2, etc)
    alg = sys.argv[2] # algoritmo que será usado para detectar as features: orb ou sift
    
    # cria o diretório de saida onde fica as imagens das correspondencias e panoramica
    create_output_dir(test_image_folder) 
   
    img_input1 = cv.imread("input/"+test_image_folder+"/img1.jpg")
    img_input2 = cv.imread("input/"+test_image_folder+"/img2.jpg")
    
    # Encontra as correspondencias    
    points_query, points_train, matches = matching(img_input1, img_input2, alg, test_image_folder)
    
    print("----------------- Info -------------------")
    
    print("Dimensoes Imagem 1")
    print("HEIGHT: ", img_input1.shape[0])
    print("WIDTH: ", img_input1.shape[1])
    
    print("Dimensoes Imagem 2")
    print("HEIGHT: ", img_input2.shape[0])
    print("WIDTH: ", img_input2.shape[1])
    
    print(f"Correspondências: {len(matches)}")
    print("------------------------------------------")
    
    # Calcular a homografia usando RANSAC
    H = ransac_calc(points_query, points_train)
    
    # salva a homografia em um arquivo txt no diretório de saida (output)
    np.savetxt("output/"+test_image_folder+"/homografy.txt", H)
    
    # Homografia T1
    # H = np.array([[1.20037934e+00, -2.23420595e-05, -6.81886838e+02],
    #       [1.50388264e-04, 1.16137777e+00, 9.62763826e+01],
    #       [-6.86162290e-08, -2.25772607e-07, 1.00000000e+00]])

    # Faz a transformação das imagens para obter a foto panorâmica usando o métod do OpenCV
    img_stitch = getPanoramicImageWithOpenCV(H, img_input1, img_input2)
    
    # Faz a transformação das imagens para obter a foto panorâmica
    # img_stitch = getPanoramicImage(H, img_input1, img_input2)
    

    # Salva a imagem no diretório de saída (output)
    cv.imwrite("output/"+test_image_folder+"/img_stitch.png", img_stitch)
    
    
if __name__ == "__main__":
   main()

