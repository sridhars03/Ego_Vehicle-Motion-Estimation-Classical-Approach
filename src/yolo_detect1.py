from ultralytics import YOLO
import cv2
import os 
import numpy as np
import time
import os
from skimage.transform import resize

# Load a model
model = YOLO("yolo11n.pt")

# filepath="G://AV projects//CV_project//cam_front//n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915244512465.jpg"
# img=cv2.imread(filepath)
# downsampled_prev = cv2.resize(img,(int(0.5*(img.shape[1])),int(0.5*(img.shape[0]))), interpolation=cv2.INTER_CUBIC)

# res=model(downsampled_prev,conf=0.5, classes=[2,7,9,10,11,12,]) #inference from pre-trained model with conf>0.5

# b=res[0].boxes

# print(len(b),type(b))
# print(b)
# print("**************")
# print(b.xyxy)
# res[0].show()


def draw_bbox_opticalflow(img, flow, bboxes, step=16):
    # Make a copy of the original image to draw on
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    # Iterate over each bounding box and process flow in each
    for box in bboxes:
        x1, y1, x2, y2 = map(int, box)  # Convert bounding box coordinates to integers
        
        # Extract flow for the bounding box region
        box_flow = flow[y1:y2, x1:x2, :]
        
        # Define a grid for drawing flow within the bounding box
        h, w = box_flow.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = box_flow[y, x].T
        
        # Offset x and y to match the bounding box location in the full image
        x += x1
        y += y1

        # Draw lines and points for each flow vector in the bounding box
        lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    
    return img_bgr

# def draw_opticalflow_fullimg(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
#     fx, fy = flow[y,x].T

#     lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)

#     img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

#     for (x1, y1), (_x2, _y2) in lines:
#         cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

#     return img_bgr

# def draw_hsv(flow):

#     h, w = flow.shape[:2]
#     fx, fy = flow[:,:,0], flow[:,:,1]

#     ang = np.arctan2(fy, fx) + np.pi
#     v = np.sqrt(fx*fx+fy*fy)

#     hsv = np.zeros((h, w, 3), np.uint8)
#     hsv[...,0] = ang*(180/np.pi/2)
#     hsv[...,1] = 255
#     hsv[...,2] = np.minimum(v*4, 255)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#     return bgr


folder_path = 'G:\AV projects\CV_project\cam_front'
image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg','.png','.jpeg'))])

'''
downsample all imgs before for loop
#store all the gray+downsampled imgs
'''
downsampled_imgs=[]
for image_file in image_files:
    img=cv2.imread(image_file)
    # downsampled_img=cv2.resize(img,(int(0.5*(img.shape[1])),int(0.5*(img.shape[0]))), interpolation=cv2.INTER_CUBIC)
    downsampled_img = resize(img, (int(0.5 * img.shape[0]), int(0.5 * img.shape[1])), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    downsampled_imgs.append(downsampled_img)

static_obj_classes,dynamic_obj_classes=[9,10,11,12,13],[0,1,2,3,5,7]

prevgray=cv2.cvtColor(downsampled_imgs[0], cv2.COLOR_BGR2GRAY)

fps=[]
for img in downsampled_imgs:
    t1=time.time()
    res=model(img,conf=0.5,classes=static_obj_classes+dynamic_obj_classes) #inference from pre-trained model with conf>0.5
    b=res[0].boxes
    detected_class=b.cls.cpu().detach().numpy() #taking all the detected classes
    
    # print(detected_class)
    # if np.isin(detected_class,static_obj_classes).any():
    #     print("******** Static obj detected *********")
        
    #     '''
    #     now do the ego car motion+pose estimation
    #     '''
    # else:
    #     print("*******Only dynamic obj *********")
    


    #doing optical flow on bboxes here for now
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    denseflow=cv2.calcOpticalFlowFarneback(prevgray, imggray, None, pyr_scale=0.33,levels=5,winsize=15,iterations=2,poly_n=7,poly_sigma=1.2,flags=0)
    # print(denseflow.shape)
    prevgray=imggray

    bboxes=b.xyxy.detach().cpu().numpy()
    fps.append(1/(time.time()-t1))

    cv2.imshow('Flow in Bounding Boxes',draw_bbox_opticalflow(imggray, denseflow, bboxes))

    cv2.waitKey(0)
    if cv2.waitKey(500)==ord('q'):
            break

cv2.destroyAllWindows()

print("time = ", sum(fps)/len(fps))
#get moving /static objs based on class id
#then get their bbox xyxy only.
#do the optical flow only in this bboxes