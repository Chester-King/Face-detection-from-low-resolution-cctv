# import the necessary packages
import numpy as np
import cv2
import os

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('i.mp4')

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
x=0
os.mkdir('detections')
while(True):
    # Capture frame-by-frame
    x+=1
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    k=0
    os.mkdir("detections/frame"+str(x))
    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        k+=1
        yA = yA+10
        yB = yB-100
        xA +=40
        xB -=50
        xC = (xA+xB)//2
        yC = (yA+yB)//2
        # cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
        cv2.circle(frame, (xC,yC), radius=27, color=(0, 0, 255), thickness=2)
        ci = frame[yC-27:yC+27,xC-27:xC+27]
        
        path = "detections/frame"+str(x)+"/"+str(k)+".png"
        cv2.imwrite(path,ci)
        
                          
    print(x)
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)