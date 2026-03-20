import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


cap = cv.VideoCapture(0)

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) # float
fps=20.0

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('saida.avi', fourcc, fps, (int(width),int(height)) )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #frame = cv.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

def read_image(imagem):
    image = cv2.imread(imagem)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def show_image(imagem):
    plt.imshow(imagem)
    plt.axis('off')
    plt.show

def white_point_correction(img, sourcewp, targetwp):
    # img escala [0, 1]
    corrected = img * (targetwp / sourcewp)
    return np.clip(corrected, 0, 1)
    
def xyz_to_rgb_gamma(xyz_img):
    # Matriz de conversão linear XYZ para RGB (exemplo para D65)
    matrix = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])

    rgb_linear = xyz_img @ matrix.T
    rgb_linear = np.clip(rgb_linear, 0, 1)

    # Correção Gama (4.1.13): sRGB standard
    # Aproximação: C = 1.055 * L^(1/2.4) - 0.055
    gamma_corrected = np.where(rgb_linear <= 0.0031308,
                               12.92 * rgb_linear,
                               1.055 * np.power(rgb_linear, 1/2.4) - 0.055)
    return gamma_corrected