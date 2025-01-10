from ultralytics import YOLO
import random
import cv2
import numpy as np
from skimage.transform import resize
from yolo_with_OF_final import *


#### model initialize here #####
model = YOLO("road_segment_yolov8.pt")

def largest_inscribed_rectangle(points,img_size):
    """
    Finds the largest axis-aligned rectangle that fits completely inside a polygon.
    """

    mask=np.zeros(img_size,dtype=np.uint8)
    height,width=mask.shape
    cv2.fillPoly(mask,[points],255)  #Fill the irregular road mask with white

    #Find the largest rectangle using histogram method
    hist=np.zeros((mask.shape[0],mask.shape[1]),dtype=np.uint8)

    #Filling Histogram:cumulative "1" in the mask for each row
    for i in range(height):
        for j in range(width):
            if mask[i,j]>0: #finding white binary pixels
                hist[i,j]=hist[i-1,j]+1 if i > 0 else 1

    # Function to find the largest rectangle in a histogram row
    def max_histogram_area(row):
        stack=[]
        max_area=0
        max_rect=(0,0,0)  #(width,height,starting column)
        row=np.append(row,0)  #Sentinel value for histogram

        for i in range(len(row)):
            while stack and row[stack[-1]] > row[i]:
                h = row[stack.pop()]
                if not stack:
                    w=i
                else:
                    w=i-stack[-1]-1

                area=h*w
                if area > max_area:
                    max_area = area
                    max_rect = (w, h, stack[-1] + 1 if stack else 0)
            stack.append(i)
        return max_area,max_rect

    #####Scan each row's histogram to get the largest rectangle#####
    max_area=0
    best_rectangle=(0,0,0,0)  #(x,y,width,height)

    for i in range(height):
        area,(rect_width,rect_height,start_col)=max_histogram_area(hist[i])
        if area>max_area:
            max_area=area
            best_rectangle=(start_col,i-rect_height+1,rect_width, rect_height)

    x, y, w, h = best_rectangle
    return {
        'area': max_area,
        'top_left': (x, y),
        'width': w,
        'height': h
    }

def road_segment(img):
    
    #####issue is here- resizing not aligning with that code ######
    # img=resize(img,(img.shape[0]//2,img.shape[1]//2),anti_aliasing=True,preserve_range=True).astype(img.dtype)

    # segment_classes=list(model.names.values())
    # print(segment_classes)

    res=model.predict(img,conf=0.4)

    road_bboxes=np.int32(res[0].boxes.xyxy.cpu().numpy())
    
    if not isinstance(res[0].masks,type(None)):
        road_mask=np.int32(res[0].masks.xy)

    else:
        ###### takes bottom part of img ######
        x_pts=np.arange(img.shape[1]//3,img.shape[1]-img.shape[1]//3)
        y_pts=np.arange(int(img.shape[0]*0.8),img.shape[0])
        top_edge = np.array([[x, y_pts[0]] for x in x_pts])
        bottom_edge = np.array([[x, y_pts[-1]] for x in x_pts])
        left_edge = np.array([[x_pts[0], y] for y in y_pts])
        right_edge = np.array([[x_pts[-1], y] for y in y_pts])

        fake_road_points = np.vstack((top_edge, right_edge, bottom_edge[::-1], left_edge[::-1]))
        road_mask=fake_road_points.reshape(1,-1,2)
        
    # cv2.polylines(img,road_mask,True,(255,0,0),1)
    # cv2.fillPoly(img,road_mask,(255,0,0))

    # cv2.imshow("road_mask",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    ########## get the biggest rectangle in this mask ########
    '''
    Get the biggest rectangle inside the road segmentation mask
    Send it to optical flow calculation - get its magnitude and direction from yolo_with_OF.py
    '''
    result=largest_inscribed_rectangle(road_mask,img_size=(img.shape[0],img.shape[1]))

    # # Display Results of rectangle algorithm
    # print(f"Largest Rectangle Area: {result['area']}")
    # print(f"Road_Rectangle Width: {result['width']}, Height: {result['height']}")
    # print(f"Road_Rectangle Top_left: {result['top_left']}")

    # img = cv2.rectangle(img,result['top_left'],(result['top_left'][0]+result['width'], result['top_left'][1]+result['height']),(0,0,255),thickness=3)

    road_mask_rectangle_bbox=[result['top_left'][0],result['top_left'][1],result['top_left'][0]+result['width'],result['top_left'][1]+result['height']]

    #### send the mask's biggest rectangle to get OF in that #####


    ###### Visualize #########
    # cv2.imshow("largest rectangle",img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return [road_mask_rectangle_bbox]
    
if __name__ =="__main__":
    img=cv2.imread("G://AV projects//CV_project//cam_front100//n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915709862465.jpg")
    road_mask_rectangle_bbox=road_segment(img)
    print(road_mask_rectangle_bbox)