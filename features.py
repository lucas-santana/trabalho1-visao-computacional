import cv2 as cv
import numpy as np

def brief_descriptor(img_path, plot=False):
    img = cv.imread(img_path)

    # Converte a imagem em escala de cinza já que BRIEF usa cálculos de intensidade de cor
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Cria o detector dos pontos de interesse
    star = cv.xfeatures2d.StarDetector_create()

    # Cria o descritor dos pontos de interesse
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

    # detectar os pontos de interesse
    kp = star.detect(img_gray, None)

    # Calcular os descritores passando a imagem e os pontos de interesse
    kp, desc = brief.compute(img_gray, kp)

    # Desenha os keypoints na imagem original com círculos verdes
    cv.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return kp, desc

def sift_descriptor(img):
    sift = cv.SIFT_create()

    # Detecta os pontos de interesse e faz o calculo dos descritores
    kp, des = sift.detectAndCompute(img, None)

    return kp, des

def orb_descriptor(img):

    orb = cv.ORB_create()

    # Detecta os pontos de interesse e faz o calculo dos descritores
    kp, des = orb.detectAndCompute(img, None)

    return kp, des

def matching(imgInput1, imgInput2, alg, output_folder):
    print("Encontrando as correspondências")
    
    # Converte as imagens para escala de cinza
    imgInput1Gray = cv.cvtColor(imgInput1, cv.COLOR_BGR2GRAY)
    imgInput2Gray = cv.cvtColor(imgInput2, cv.COLOR_BGR2GRAY)
    
    if alg == "sift":
        kp1, des1 = sift_descriptor(imgInput1Gray)
        kp2, des2 = sift_descriptor(imgInput2Gray)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        
    elif alg == "orb":
        kp1, des1 = orb_descriptor(imgInput1Gray)
        kp2, des2 = orb_descriptor(imgInput2Gray)
        
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
    
    # Ordena as correpondências pela distância para desenha as melhores
    matches = sorted(matches, key = lambda x:x.distance)
    img = cv.drawMatches(imgInput1, kp1, imgInput2, kp2, matches[:50], None,flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("output/"+output_folder+"/matches.png", img)
        
    """
        obter os keypoints dos melhores matches
        para cada match m, pego pt do keypoint na posição m.qyeryIdx
    """
    points_query = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points_train = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    

    return points_query, points_train, matches



    