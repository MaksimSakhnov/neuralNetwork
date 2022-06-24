import numpy as np
import cv2
import random

print("0 - тренировочные данные, 1 - тестовые")
var = int(input())
if(var==0):

    for item in range(10001):
        temp = int(random.random() * 100) + 1

        answer = ""
        if(temp > 0 and temp <= 10):
            answer = "zero"
        elif(temp > 10 and temp <=20):
            answer = "one"
        elif (temp > 20 and temp <= 30):
            answer = "two"
        elif (temp > 30 and temp <= 40):
            answer = "three"
        elif (temp > 40 and temp <= 50):
            answer = "four"
        elif (temp > 50 and temp <= 60):
            answer = "five"
        elif (temp > 60 and temp <= 70):
            answer = "six"
        elif (temp > 70 and temp <= 80):
            answer = "seven"
        elif (temp > 80 and temp <= 90):
            answer = "eight"
        elif (temp > 90 and temp <= 100):
            answer = "nine"

        FILE = "train/%s.png" %(str(temp))
        rez = open("train/train.txt", 'a')
        color_image = cv2.imread(FILE)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        n_image = np.divide(gray_image, 255.0)

        for i in range(len(n_image)):
            for j in range (len(n_image[i])):
                n_image[i][j] = abs(1 - n_image[i][j]) * 100 // 10 / 10

        for i in range(len(n_image)):
            temp = n_image[i]
            for j in range (len(temp)):
                rez.write(str(temp[j]))
                rez.write(" ")
            rez.write("\n")
        rez.write(answer)
        rez.write("\n")
        rez.close()
elif(var==1):
    for item in range(1, 21):
        FILE = "test/%s.png" % (item)
        rez = open("test/test.txt", 'a')
        color_image = cv2.imread(FILE)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        n_image = np.divide(gray_image, 255.0)

        for i in range(len(n_image)):
            for j in range(len(n_image[i])):
                n_image[i][j] = abs(1 - n_image[i][j]) * 100 // 10 / 10

        for i in range(len(n_image)):
            temp = n_image[i]
            for j in range(len(temp)):
                rez.write(str(temp[j]))
                rez.write(" ")
            rez.write("\n")

        rez.write("\n")
        rez.close()
