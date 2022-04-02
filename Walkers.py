import cv2


# Create our body classifier

classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    ret,frame = cap.read()
    greyimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(greyimg, 1.1,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (w+x,h+y), (0,0,255), 3)
    cv2.imshow('live', frame)
    if cv2.waitKey(25)==32:
        break


cap.release()
cv2.destroyAllWindows()
