import cv2
import numpy as np

def calculate_angle_v2(x, y, ellipse):
    h, k = ellipse[0]
    a, b = max(ellipse[1]) / 2,min(ellipse[1]) / 2
    phi = np.radians(90-ellipse[2])

    numerator = (x - h) * np.cos(phi) + (y - k) * np.sin(phi)
    denominator = (x - h) * np.sin(phi) - (y - k) * np.cos(phi)
    slope_tangent = -numerator / denominator * (b**2 / a**2)
    angle =np.degrees(np.arctan(slope_tangent))
    if angle < 0:
        angle += 180
    if x>h:
        angle = 180-angle
    return angle

def find_contact_angle(image,rectI):
    # Read the image
    # Convert the image to grayscale
    crop_img=image[rectI.outRect.y:rectI.outRect.y+rectI.outRect.h,rectI.outRect.x:rectI.outRect.x+rectI.outRect.w]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (rectI.blur_size, rectI.blur_size), 0)

    # Use Canny edge detector to find edges
    edges = cv2.Canny(blurred, rectI.canny1, rectI.canny2,apertureSize=rectI.apertureSize,L2gradient=rectI.L2gradient)
    #cv2_imshow(edges)
    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the droplet, using contour perimeter as the key
    largest_contour = max(contours, key=lambda x: cv2.arcLength(x, closed=True))
    ellipse = cv2.fitEllipse(largest_contour)

    cv2.ellipse(crop_img, ellipse, (255, 255, 0), 2)
    cv2.drawContours(crop_img, [largest_contour], 0, (0,255,0), 3)
    left_of_center = np.array([point for point in largest_contour[:, 0] if point[0] < crop_img.shape[1]/2])
    bottom_most_left = max(left_of_center, key=lambda x: x[1])

    right_of_center = np.array([point for point in largest_contour[:, 0] if point[0] > crop_img.shape[1]/2])
    bottom_most_right = max(right_of_center, key=lambda x: x[1])

    cv2.circle(crop_img,bottom_most_left,5,(0,0,255),cv2.FILLED)
    cv2.circle(crop_img,bottom_most_right,5,(0,0,255),cv2.FILLED)
    x_l,y_l=bottom_most_left
    a_l = calculate_angle_v2(x_l,y_l,ellipse)

    x_r,y_r=bottom_most_right
    a_r = calculate_angle_v2(x_r,y_r,ellipse)

    line_length = 100

    # Calculate the endpoint of the line
    image[rectI.outRect.y:rectI.outRect.y+rectI.outRect.h,rectI.outRect.x:rectI.outRect.x+rectI.outRect.w]=crop_img

    end_point_l = (int(x_l+rectI.outRect.x + line_length * np.cos(-np.radians(a_l))),
                int(y_l+rectI.outRect.y + line_length * np.sin(-np.radians(a_l))))
    cv2.line(image, [x_l+rectI.outRect.x,y_l+rectI.outRect.y], end_point_l, (0, 255, 255), 2)
    end_point_r = (int(x_r+rectI.outRect.x - line_length * np.cos(np.radians(a_r))),
                int(y_r+rectI.outRect.y + line_length * np.sin(-np.radians(a_r))))
    cv2.line(image, [x_r+rectI.outRect.x,y_r+rectI.outRect.y], end_point_r, (0, 255, 255), 2)
    #print(a_l,a_r)

    # Display the result

    # Concatenate the original frame and processed frame horizontally

    return a_l,a_r
