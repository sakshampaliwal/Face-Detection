import cv2

#importing face data
trained_data=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#capturing image from webcam:
webcam=cv2.VideoCapture(0) #You can also use video by just putting video file location in this function

#iterate over the frame:
while True:
    ret,frame=webcam.read()#ret contains true or false value and frame contains the frame of the video(webcam video)
    
    # converting into grayscale_image:
    grayscale_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #getting face coordinates:
    face_coordinates=trained_data.detectMultiScale(grayscale_image)

    #rectangles around faces:
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,256,0),2) #here it is bgr value where we have only included green value and '2' is the boldness of the line

    cv2.imshow('Laptop_Webcam',frame)
    x=cv2.waitKey(1)

    #breaking the loop using q button ASCII Code of Q=81 and q=113
    if x==113 or x==81:
        break

#Releasing the video capture object
webcam.release()

print("code completed")
