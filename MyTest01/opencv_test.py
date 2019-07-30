import cv2

pic = cv2.imread("pic3.jpg")
cv2.line(pic, (170, 270), (300, 400), (234, 213, 208), 2)
cv2.rectangle(pic, (150, 200), (200, 300), (83, 127, 134), 2)
cv2.circle(pic, (350, 450), 50, (34, 56, 23), 2)
cv2.imshow("pic", pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
