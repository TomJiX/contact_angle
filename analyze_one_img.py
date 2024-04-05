import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle_v2(x, y, ellipse):
    h, k = ellipse[0]
    a, b = max(ellipse[1]) / 2, min(ellipse[1]) / 2
    phi = np.radians(90 - ellipse[2])

    numerator = (x - h) * np.cos(phi) + (y - k) * np.sin(phi)
    denominator = (x - h) * np.sin(phi) - (y - k) * np.cos(phi)
    slope_tangent = -numerator / denominator * (b**2 / a**2)
    angle = np.degrees(np.arctan(slope_tangent))
    
    if angle < 0:
        angle += 180
    
    if x<h:
        angle=180-angle
    return angle

def edge_detection(image, rectI):
    blue_threshold = rectI.blue_thresh
    blue_mask = image[:, :, 0] > blue_threshold
    copy = np.copy(image)
    copy[blue_mask] = [255, 255, 255]
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (rectI.blur_size, rectI.blur_size), 0)
    
    # Perform edge detection on the whole image
    edges = cv2.Canny(blurred, rectI.canny1, rectI.canny2, apertureSize=rectI.apertureSize, L2gradient=rectI.L2gradient)
    rectI.edges = edges
    return 

def find_best_ellipse(largest_contour,limit=0.7,step=20):
    limit = int(limit*largest_contour.shape[0])
    min_distance = 10000
    ellipse = None
    new_largest_contour = largest_contour
    cnt=np.squeeze(largest_contour)

    ellipse = cv2.fitEllipse(cnt)
    contour = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                            (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                            int(ellipse[2]), 0, 360, 10)
    order=[]
    for l,pt in enumerate(cnt):
        pt = (int(pt[0]), int(pt[1]))
        d = cv2.pointPolygonTest(contour, pt , True)
        order.append((d,l))
    order=sorted(order,key=lambda x:np.abs(x[0]))
    all_cnt=[]
    tmp_cnt=cnt
    out_ellipse=[]
    #np.abs(distance)<np.abs(max_distance) and
    while tmp_cnt.shape[0]>limit:
        
        tmp_cnt = np.delete(cnt, [x[1] for x in order[-step:]], axis=0)
        step+=step
        all_cnt.append(tmp_cnt)
        
        ellipse = cv2.fitEllipse(tmp_cnt)
        
        contour = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                                int(ellipse[2]), 0, 360, 10)
        distance = 0
        for l,pt in enumerate(tmp_cnt):
            pt = (int(pt[0]), int(pt[1]))
            d = cv2.pointPolygonTest(contour, pt , True)
            distance+=np.abs(d)
        if distance<min_distance:
            min_distance=distance
            new_largest_contour=tmp_cnt
            out_ellipse=ellipse
                
    return out_ellipse,new_largest_contour

def find_contact_angle(image, rectI):
    scale_features = int(0.002 * image.shape[0])
    crop_img = image[rectI.outRect.y:rectI.outRect.y + rectI.outRect.h,
                     rectI.outRect.x:rectI.outRect.x + rectI.outRect.w]
    
    edges = rectI.edges
    
    edges_crop = edges[rectI.outRect.y:rectI.outRect.y + rectI.outRect.h,
                          rectI.outRect.x:rectI.outRect.x + rectI.outRect.w]
    
    contours_inside_crop, _ = cv2.findContours(edges_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours_inside_crop, key=lambda x: cv2.arcLength(x, closed=True))

    print("new loop")
    ellipse,c = find_best_ellipse(largest_contour)
    cv2.ellipse(crop_img, ellipse, (255, 255, 0), 2 * scale_features)
    cv2.drawContours(crop_img, [c], -1, (0, 255, 0), 5)
    
    left,right=find_end_point(c,crop_img)

    cv2.circle(crop_img, left, 5 * scale_features, (0, 0, 255), cv2.FILLED)
    cv2.circle(crop_img, right, 5 * scale_features, (0, 0, 255), cv2.FILLED)
    
    x_l, y_l = left
    a_l = calculate_angle_v2(x_l, y_l, ellipse)
    x_r, y_r = right
    a_r = calculate_angle_v2(x_r, y_r, ellipse)
    
    
    line_length = 100 * scale_features
    image[rectI.outRect.y:rectI.outRect.y+rectI.outRect.h,rectI.outRect.x:rectI.outRect.x+rectI.outRect.w]=crop_img

    end_point_l = (int(x_l + rectI.outRect.x - line_length * np.cos(-np.radians(a_l))),
                   int(y_l + rectI.outRect.y + line_length * np.sin(-np.radians(a_l))))
    cv2.line(image, [x_l + rectI.outRect.x, y_l + rectI.outRect.y], end_point_l, (0, 255, 255), 2 * scale_features)
    
    end_point_r = (int(x_r + rectI.outRect.x + line_length * np.cos(np.radians(a_r))),
                   int(y_r + rectI.outRect.y + line_length * np.sin(-np.radians(a_r))))
    cv2.line(image, [x_r + rectI.outRect.x, y_r + rectI.outRect.y], end_point_r, (0, 255, 255), 2 * scale_features)
    
    return a_l, a_r


def find_end_point(largest_contour,crop_img):
    left_of_center = np.array([point for point in largest_contour if point[0] < crop_img.shape[1]/2])
    left_point = max(left_of_center, key=lambda x: x[1])
    right_of_center = np.array([point for point in largest_contour if point[0] > crop_img.shape[1]/2])
    right_point = max(right_of_center, key=lambda x: x[1])
    return left_point,right_point
