
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
    test_image_folder = sys.argv[1]
    alg = sys.argv[2]
    
    create_output_dir(test_image_folder)
   
    img_input1 = cv.imread("input/"+test_image_folder+"/img1.jpg")
    img_input2 = cv.imread("input/"+test_image_folder+"/img2.jpg")
        
    points_query, points_train, matches = matching(img_input1, img_input2, alg, test_image_folder)
    
    print("----------------- Info -------------------")
    
    print("Dimensoes Imagem 1")
    print("HEIGHT: ", img_input1.shape[0])
    print("WIDTH: ", img_input1.shape[1])
    
    print("Dimensoes Imagem 2")
    print("HEIGHT: ", img_input2.shape[0])
    print("WIDTH: ", img_input2.shape[1])
    
    print(f"Correspondências: {len(points_query)}")
    print("------------------------------------------")
    
    H = ransac_calc(points_query, points_train)
    
    np.savetxt("output/"+test_image_folder+"/homografy.txt", H)
    
    # Homografia T1
    # H = np.array([[1.20037934e+00, -2.23420595e-05, -6.81886838e+02],
    #       [1.50388264e-04, 1.16137777e+00, 9.62763826e+01],
    #       [-6.86162290e-08, -2.25772607e-07, 1.00000000e+00]])

    img_stitch = getPanoramicImageWithOpenCV(H, img_input1, img_input2)
    
    # img_stitch = getPanoramicImage(H, imgInput1, imgInput2)
    cv.imwrite("output/"+test_image_folder+"/img_stitch.png", img_stitch)
    
    
if __name__ == "__main__":
   main()

