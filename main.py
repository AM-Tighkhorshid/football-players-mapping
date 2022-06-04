import cv2
import numpy as np
from keras.models import load_model
from tracker import *
import time

# Create tracker object
tracker = EuclideanDistTracker()


cap = cv2.VideoCapture("./data/output.mp4")
img = cv2.imread("./data/2D_field.png")



#subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
subtractor = cv2.createBackgroundSubtractorMOG2()

######################################################################################################################

p1 = (872,780)
p2 = (640,110)
p3 = (1140,117)
p4 = (139,168)
p5 = (139,168)

points1 = np.array([p1,p2,p3,p4], dtype=np.float32)

p1 = (525,700)
p2 = (525,0)
p3 = (886,143)
p4 = (164,143)
p5 = (164,143)

points2 = np.array([p1,p2,p3,p4], dtype=np.float32)

H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0)

############################################################################################################################
targe_size = 32
model = load_model('model3.h5')

tic = time.time()
count = 0

while True:
    img = cv2.imread("./data/2D_field.png")
    ret, frame = cap.read()
    if ret == False:
        break
    count = count + 1
    #frame = cv2.GaussianBlur(frame,(3,3),1)
    mask = subtractor.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((8,2),np.uint8)
    kernelclose = np.ones((4,8),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    name = 0
    detections = []
    playerCordinates = []
    images = np.zeros(shape=(1,32,32,3))
    for cnt in contours:
        
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            
            if y < 100 or y > 680:
                continue
            
            detections.append([x, y, w, h])
            
            #playerCordinate = cv2.perspectiveTransform(H,pts).reshape(1,2)
            
            
            point = [[x+w/2],
                     [y+h],
                     [1]]
            
            playerCordinate = np.matmul(H, point)
            
            playerCordinate = playerCordinate/playerCordinate[2]
            
            # img = cv2.circle(img,(int(playerCordinate[0]), int(playerCordinate[1])),5,(0, 0, 255),thickness=5)
            
            
            playerCordinates.append(playerCordinate)
            
            
            dest_points = np.array([(0,0),(targe_size,0),(targe_size,targe_size),(0,targe_size)], dtype=np.float32)
            thickness = 1
            sourc_points = np.array([(x,y), (x + w,y), (x+w,y+h), (x,y+h)], dtype=np.float32)
            H1 = cv2.getPerspectiveTransform(sourc_points, dest_points)
            pic = cv2.warpPerspective(frame,H1,  (targe_size, targe_size))
            pic = np.expand_dims(pic,axis=0)
            images = np.concatenate((images,pic),axis=0)
            # cv2.imshow("11",images[name])          
            name += 1
            
            # y_pred = model.predict(np.expand_dims(pic,axis=0))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
            
            # if np.argmax(y_pred) == 1:
            #     img = cv2.circle(img,(int(playerCordinate[0]), int(playerCordinate[1])),5,(0, 0, 255),thickness=5)
            # elif np.argmax(y_pred) == 2:
            #     img = cv2.circle(img,(int(playerCordinate[0]), int(playerCordinates[1])),5,(255, 0, 0),thickness=5)
            
    #################################################### classification with CNN
    if (images.shape[0] != 0):
        y_pred = model.predict(images)
    for i in range(len(playerCordinates)):
        if np.argmax(y_pred[i]) == 1:
            img = cv2.circle(img,(int(playerCordinates[i][0]), int(playerCordinates[i][1])),5,(255, 0, 0),thickness=5)
        elif np.argmax(y_pred[i]) == 2 or np.argmax(y_pred[i]) == 0:
            img = cv2.circle(img,(int(playerCordinates[i][0]), int(playerCordinates[i][1])),5,(0, 0, 255),thickness=5)
    
    
    ########################################################### tracker part
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        point = [[x+w/2],
                     [y+h],
                     [1]]
        playerCord = np.matmul(H, point)
        playerCord = playerCord/playerCord[2]
        cv2.putText(img, str(id), (int(playerCord[0]), int(playerCord[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("img", img)

    
    key = cv2.waitKey(1)
    if key == 27:
        break
toc = time.time()

time_ratio = (toc-tic)/(count)

print("fps = " + str(1/time_ratio))
cap.release()
cv2.destroyAllWindows()
