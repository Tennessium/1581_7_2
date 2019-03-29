import cv2 as cv
from random import randint
import numpy as np
import Parameters as ps  # Импортируем вручную подобранную маску
from tqdm import tqdm
import argparse


def get_frame(path):  # Получение изображения
    img = cv.imread(path)
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (5, 5))
    return hsv


def set_bounds(path):  # Получение характеристик желаемого прямоугольника
    img = cv.imread(path)
    while (True):
        newframe = img
        newframe = cv.resize(newframe, (800, 600), interpolation=cv.INTER_AREA)
        x = cv.getTrackbarPos("x", "Sliders")
        x1 = cv.getTrackbarPos("x1", "Sliders")
        y = cv.getTrackbarPos("y", "Sliders")
        y1 = cv.getTrackbarPos("y1", "Sliders")
        cv.rectangle(newframe, (x, y), (x1, y1), (0, 255, 255), 2)
        cv.imshow('bounds', newframe)
        if cv.waitKey(1) == 27:
            break
    return abs(x - x1), abs(y - y1)


def get_tg(HSV, frame):  # Вычисление оцениваемого тангенса
    mask = cv.inRange(frame, HSV[0], HSV[1])
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = contours_info[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contour = contours[0]
    (x, y, w, h) = cv.boundingRect(contour)
    tg = abs(w) / abs(h)
    return tg


def empty(a):
    pass


def error(HSV):  # Получение ошибки
    global roi1
    global roi2
    global frame1
    global frame2
    tg1, tg2 = (roi1[0] / roi1[1]), (roi2[0] / roi2[1])
    tg1_new, tg2_new = get_tg(HSV, frame1), get_tg(HSV, frame2)
    error = abs(abs(tg1 - tg1_new) / tg1 - abs(tg2 - tg2_new) / tg2)
    return error


parser = argparse.ArgumentParser()  # Парсинг аргументов из командной строки
parser.add_argument("--path")
parser.add_argument("--epsilon")
parser.add_argument("--step")
parser.add_argument("--skip")
args = parser.parse_args()

if not args.path or not args.epsilon or not args.step:
    print("Not enough arguments")
    quit()
if args.skip:  # Шаг по списку сгенерированных масок
    skip = int(args.skip)
else:
    skip = 1
path = args.path  # Путь до директории с изображениями
epsilon = int(args.epsilon)  # Окрестность, в которой будут подбираться параметры маски
step = int(args.step)  # Шаг по окрестности

try:
    open(path + "1.jpg")
    open(path + "2.jpg")
except FileNotFoundError:
    print("Images not found")
    quit()

img1 = path + "1.jpg"  # Путь до первой картинки
img2 = path + "2.jpg"  # Путь до второй картинки
frame1, frame2 = get_frame(img1), get_frame(img2)
masks = np.array([[[0, 0, 0], [0, 0, 0]]])  # Инициализация списка для перебора
(h0, s0, v0), (H0, S0, V0) = ps.mask_HSV
print("Creating masks map...")
for h in tqdm(range(h0 - epsilon, h0 + epsilon + 1, step)):  # Генерация списка для перебора
    if h > 0:
        for s in range(s0 - epsilon, s0 + epsilon + 1, step):
            if s > 0:
                for v in range(v0 - epsilon, v0 + epsilon + 1, step):
                    if v > 0:
                        for H in range(H0 - epsilon, H0 + epsilon + 1, step):
                            if H > 0:
                                for S in range(S0 - epsilon, S0 + epsilon + 1, step):
                                    if S > 0:
                                        for V in range(V0 - epsilon, V0 + epsilon + 1, step):
                                            if V > 0:
                                                masks = np.concatenate((masks, [[[h, s, v], [H, S, V]]]), axis=0)

masks = masks[1:]
print("Done, masks created: " + str(len(masks)))

cv.namedWindow("Sliders")  # Создание окна со слайдерами
cv.createTrackbar("x", "Sliders", 0, 800, empty)
cv.createTrackbar("x1", "Sliders", 0, 800, empty)
cv.createTrackbar("y", "Sliders", 0, 600, empty)
cv.createTrackbar("y1", "Sliders", 0, 600, empty)

roi1 = set_bounds(img1)  # Задание желаемых прямоугольников
roi2 = set_bounds(img2)
cv.destroyAllWindows()
print("Current mask error: " + str(round(error(ps.mask_HSV) * 100, 2)) + "%")
print("Starting calculation...")
NewOptimalMask = ps.mask_HSV
for mask in tqdm(masks[::skip]):  # Поиск маски с наименьшей ошибкой
    if error(mask) < error(NewOptimalMask):
        NewOptimalMask = mask
print("New optimal mask: ")
print(tuple(NewOptimalMask[0]), tuple(NewOptimalMask[1]), sep=", ")
print("New mask error: " + str(round(error(NewOptimalMask) * 100, 2)) + "%")
# Подсчет нового среднего значения тангенса
print("New approximated tangent:  " + str(
    (get_tg(NewOptimalMask, get_frame(img1)) + get_tg(NewOptimalMask, get_frame(img2))) / 2))

cv.destroyAllWindows()
