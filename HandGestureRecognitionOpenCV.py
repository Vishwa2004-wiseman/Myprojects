import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)  # Change to 1 if you have a secondary camera

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Draw a rectangle in the original image to mark the region of interest (ROI)
    cv2.rectangle(img, (50, 50), (400, 400), (0, 255, 0), 2)
    crop_img = img[50:400, 50:400]  # Extract the ROI

    # Convert to grayscale and apply GaussianBlur
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # Apply thresholding to get binary image
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    # Find contours
    contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours:
        # Get the largest contour
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Create a convex hull around the largest contour
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # Calculate the length of all sides of the triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                # Apply the cosine rule to calculate the angle
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # If the angle is less than 90 degrees, it's a defect
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_img, far, 1, [0, 0, 255], -1)

                # Draw lines around the defect points
                cv2.line(crop_img, start, end, [0, 255, 0], 2)

            # Recognize gestures based on the number of defects
            if count_defects == 1:
                cv2.putText(img, "GESTURE ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 2:
                cv2.putText(img, "GESTURE TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 3:
                cv2.putText(img, "GESTURE THREE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif count_defects == 4:
                cv2.putText(img, "GESTURE FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Hello World!!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # Show the results
            cv2.imshow('Gesture', img)
            all_img = np.hstack((crop_img, cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)))
            cv2.imshow('Contours', all_img)
        else:
            print("No convexity defects found.")
    else:
        print("No contours found.")

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
