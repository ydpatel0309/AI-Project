import cv2

# initialize video capture
cap = cv2.VideoCapture(0)

# initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# set minimum threshold value for foreground detection
THRESHOLD = 100

# set minimum contour area to filter out small movements
MIN_CONTOUR_AREA = 1000

while True:
    # read a frame from the video stream
    ret, frame = cap.read()

    # apply background subtraction to detect motion
    fgmask = fgbg.apply(frame)

    # apply thresholding to clean up the motion mask
    thresh = cv2.threshold(fgmask, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw bounding boxes around the contours
    for contour in contours:
        # get contour area
        area = cv2.contourArea(contour)
        # only draw bounding box if contour area is greater than MIN_CONTOUR_AREA
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display the resulting image
    cv2.imshow('Motion Detection', frame)

    # check for escape key press to exit
    if cv2.waitKey(1) == 27:
        break

# release video capture and close windows
cap.release()
cv2.destroyAllWindows()
