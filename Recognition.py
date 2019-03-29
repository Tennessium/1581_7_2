import cv2 as cv
import Parameters as ps
import argparse


def recognition(frame):  # Распознавание шестерней
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Переход в HSV
    hsv = cv.blur(hsv, (5, 5))  # Размытие
    mask = cv.inRange(hsv, ps.mask_specified[0], ps.mask_specified[1])  # Создание маски
    contours_info = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # Поиск контуров
    contours = contours_info[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    (x, y, w, h) = cv.boundingRect(contours[0])  # Выявление прямоугльника по контурам
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    tan = w / h  # Вычисление тангенса
    if tan > ps.approx_tangent:  # Сравнение тангенса с заранее вычисленным средним значением
        cv.putText(frame, 'Shesternia-1', (x, y - 5), font, 2, (0, 0, 0), 2, cv.LINE_AA)  # Отрисовка надписи
        print("sh-1")
    else:
        cv.putText(frame, 'Shesternia-2', (x, y - 5), font, 2, (0, 0, 0), 2, cv.LINE_AA)
        print("sh-2")
    return frame


def get_frame(path):  # Получение изображения
    img = cv.imread(path)
    frame = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    return frame


font = cv.QT_FONT_NORMAL  # Задание шрифта

parser = argparse.ArgumentParser()  # Парсинг аргумента из командной строки
parser.add_argument("--path")
args = parser.parse_args()
if not args.path:
    print("Not enough arguments")
    quit()
try:
    open(args.path + "1.jpg")
    open(args.path + "2.jpg")
except FileNotFoundError:
    print("Images not found")
    quit()

img1 = get_frame(args.path + "1.jpg")  # Получение изображений
img2 = get_frame(args.path + "2.jpg")

cv.imshow("1", img1)  # Отображение изначальных изображений
cv.imshow("2", img2)
cv.waitKey(2000)
cv.imshow("1", recognition(img1))  # Отображений изображений с распознанными шестернями
cv.imshow("2", recognition(img2))

while True:
    if cv.waitKey() == 27:
        break
cv.destroyAllWindows()
