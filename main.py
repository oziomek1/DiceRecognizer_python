import numpy as np
import cv2

def fun(x):
    pass


cap = cv2.VideoCapture(1)
windowname = 'DiceRecognizer'
cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('threshold', windowname, 0, 255, fun)
cv2.createTrackbar('canny_1', windowname, 0, 10, fun)
cv2.createTrackbar('canny_2', windowname, 0, 25, fun)

# Capture first frame without dice
i = 0
# Loop to secure that backfroundFrame is not blank
while i < 10:
    backgroundRet, backgroundFrame = cap.read()
    i +=1
# Change backgroundFrame to grayscale
backgroundFrame = cv2.cvtColor(backgroundFrame, cv2.COLOR_BGR2GRAY)

while True:
    # Capture video frame-by-frame
    ret, originalFrame = cap.read()

    # Change originalFrame to grayscale
    processedFrame = cv2.cvtColor(originalFrame, cv2.COLOR_BGR2GRAY);

    # Difference between two frame
    # processedFrame = cv2.absdiff(processedFrame, backgroundFrame)

    # Applying image segmantation using threshold
    threshold_value = cv2.getTrackbarPos('threshold', windowname)
    ret, processedFrame = cv2.threshold(processedFrame, threshold_value, 255, cv2.THRESH_OTSU or cv2.THRESH_BINARY)

    # Apply edge detection using canny algorithm (optimal detector)
    cannythreshold1 = cv2.getTrackbarPos('canny_1', windowname)
    cannythreshold2 = cv2.getTrackbarPos('canny_2', windowname)
    cv2.Canny(processedFrame, cannythreshold1, cannythreshold2, processedFrame, 3, False)

    # Hough transform to find dots
    circles = cv2.HoughCircles(processedFrame, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=100, param2=30, minRadius=5, maxRadius=25)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(originalFrame, (x, y), r, (0, 165, 255), 3)
            circle_text = 'val: (%d:%d, r=%d' % (x, y, r)
            cv2.putText(originalFrame, circle_text, (x, y + r + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, 8)

    # FindContours to find contours of dice
    processedFrame, contours, hierarchy = cv2.findContours(processedFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            print('area')
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(originalFrame, contours, -1, (255, 0, 0), 3)
            # cv2.rectangle(originalFrame, (x, y), (x + w, y + h), (255, 0,0), 5)
            rectangle_text = 'val: %d (%d:%d) (%d:%d)' % (w*h, x, y, x+w, y+h)
            cv2.putText(originalFrame, rectangle_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, 8)

    # Display the results
    cv2.imshow(windowname, originalFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capturing video
cap.release()
cv2.destroyAllWindows()
