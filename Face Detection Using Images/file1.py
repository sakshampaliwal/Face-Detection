import cv2

#Our Trained data:
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#original image read:
original_image=cv2.imread('E:\Python Projects\Face Detection\GroupPic\comm1.jpg')

#converting to grayscale:
grayscale_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)

#Detect Face Coordinates:
face_coordinates=trained_face_data.detectMultiScale(grayscale_image)

#rectangle around faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,256,0),2)

#showing the image now:
cv2.imshow('Image Uploaded',original_image)
cv2.waitKey()
print("code completed")