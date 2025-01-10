from ultralytics import YOLO
import cv2
import os 
import numpy as np
import time
import os
from skimage.transform import resize
import road_segmentation
import of_2_world_frame

####### Optical flow in individual bounding boxes visualization ####### 
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
################################


############ calc average motion direction from each bbox ##############
def get_flow_vector(bbox_flow_list):
    motion_dir_list,flow_mag_list=[],[]
    for bbox_flow in bbox_flow_list:
        fx,fy=bbox_flow[...,0],bbox_flow[...,1]
        avg_fx,avg_fy=np.mean(fx),np.mean(fy)
        mag_flow=np.sqrt(avg_fx**2+avg_fy**2)

        if mag_flow<1e-4:
            motion_dir_list.appenqd((0,0))
            flow_mag_list.append(0)
        else:
            motion_dir_list.append((avg_fx/mag_flow,avg_fy/mag_flow))
            flow_mag_list.append(mag_flow)
        
    motion_dir_list=np.round(motion_dir_list,4)
    flow_mag_list=np.round(flow_mag_list,4)

    return motion_dir_list,flow_mag_list
##############################################



########### Visualize Optical Flow in the entire image ##########
def draw_opticalflow_fullimg(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr
########################################


########### Draw HSV based on OF values ###############
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
#####################################################


def main():
    #Loading the YOLO pretrained model
    # model = YOLO("yolo11n.pt")
    model = YOLO("yolo11n-seg.pt")
    folder_pathF = 'G:\AV projects\CV_project\cam_front100'
    # folder_pathFL = 'G:\AV projects\CV_project\cam_front_left100'
    # folder_pathFR = 'G:\AV projects\CV_project\cam_front_right100'
    # folder_pathB = 'G:\AV projects\CV_project\cam_back100'
    # folder_pathBL = 'G:\AV projects\CV_project\cam_back_left100'
    # folder_pathBR = 'G:\AV projects\CV_project\cam_back_right100'
    image_filesF = sorted([os.path.join(folder_pathF, f) for f in os.listdir(folder_pathF) if f.endswith(('.jpg','.png','.jpeg'))])
    # image_filesFL = sorted([os.path.join(folder_pathFL, f) for f in os.listdir(folder_pathFL) if f.endswith(('.jpg','.png','.jpeg'))])
    # image_filesFR = sorted([os.path.join(folder_pathFR, f) for f in os.listdir(folder_pathFR) if f.endswith(('.jpg','.png','.jpeg'))])
    # image_filesB = sorted([os.path.join(folder_pathB, f) for f in os.listdir(folder_pathB) if f.endswith(('.jpg','.png','.jpeg'))])
    # image_filesBL = sorted([os.path.join(folder_pathBL, f) for f in os.listdir(folder_pathBL) if f.endswith(('.jpg','.png','.jpeg'))])
    # image_filesBR = sorted([os.path.join(folder_pathBR, f) for f in os.listdir(folder_pathBR) if f.endswith(('.jpg','.png','.jpeg'))])
    

    '''
    downsample all imgs before for loop
    #store all the gray+downsampled imgs
    '''

    downsampled_imgsF,downsampled_grayF=[],[]
    # downsampled_imgsFL,downsampled_grayFL=[],[]
    # downsampled_imgsFR,downsampled_grayFR=[],[]
    # downsampled_imgsB,downsampled_grayB=[],[]
    # downsampled_imgsBL,downsampled_grayBL=[],[]
    # downsampled_imgsBR,downsampled_grayBR=[],[]
    
    for image_file in image_filesF:
        img=cv2.imread(image_file)
        downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
        downsampled_imgsF.append(downsampled_img)
        downsampled_grayF.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))

    # for image_file in image_filesFL:
    #     img=cv2.imread(image_file)
    #     downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    #     downsampled_imgsFL.append(downsampled_img)
    #     downsampled_grayFL.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))
    
    # for image_file in image_filesFR:
    #     img=cv2.imread(image_file)
    #     downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    #     downsampled_imgsFR.append(downsampled_img)
    #     downsampled_grayFR.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))
    
    # for image_file in image_filesB:
    #     img=cv2.imread(image_file)
    #     downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    #     downsampled_imgsB.append(downsampled_img)
    #     downsampled_grayB.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))
    
    # for image_file in image_filesBL:
    #     img=cv2.imread(image_file)
    #     downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    #     downsampled_imgsBL.append(downsampled_img)
    #     downsampled_grayBL.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))
    
    # for image_file in image_filesBR:
    #     img=cv2.imread(image_file)
    #     downsampled_img = resize(img, (img.shape[0]//2,img.shape[1]//2), anti_aliasing=True, preserve_range=True).astype(img.dtype)
    #     downsampled_imgsBR.append(downsampled_img)
    #     downsampled_grayBR.append(cv2.cvtColor(downsampled_img,cv2.COLOR_BGR2GRAY))


    #define the classes to detect
    static_obj_classes,dynamic_obj_classes=[9,10,11,12,13],[0,1,2,3,5,7]

    
    ########## continue from this for 6 cams ##########
    prevgray=cv2.cvtColor(downsampled_imgsF[0], cv2.COLOR_BGR2GRAY)

    count=0
    for img in downsampled_imgsF[1:]:
        res=model(img,conf=0.51,classes=static_obj_classes+dynamic_obj_classes) #inference from pre-trained model with conf>0.5
        b=res[0].boxes
        
        ####################
        # detected_class=b.cls.cpu().detach().numpy() #taking all the detected classes
        # print(detected_class)
        
        # if np.isin(detected_class,static_obj_classes).any():
        #     print("******** Static obj detected *********")
            
        #     '''
        #     now do the ego car motion+pose estimation
        #     '''
        # else:
        #     print("*******Only dynamic obj *********")
        ####################

        imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        denseflow_fullimg=cv2.calcOpticalFlowFarneback(prevgray, imggray, None, pyr_scale=0.33,levels=5,winsize=15,iterations=2,poly_n=7,poly_sigma=1.2,flags=0)
        # print(denseflow_fullimg.shape)
        prevgray=imggray

        bboxes=b.xyxy.detach().cpu().numpy()
        # print(bboxes.shape)

        #### call road_segmentation-get its bbox #########
        road_mask_rectangle_bbox=road_segmentation.road_segment(img)
        # print(road_mask_rectangle_bbox)

        ##### the road's bbox is the last bbox ######
        bboxes=np.concatenate((bboxes,road_mask_rectangle_bbox))
        # print(bboxes)

        ########## get OF in each bbox ###############
        bbox_flow_list=[]  #this ll be the list of OF in bounding boxes from YOLO detection
        for box in bboxes:
            bbox_flow=denseflow_fullimg[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
            bbox_flow_list.append(bbox_flow)
        

        ###### Calculate direction/magnitude of OF #########
        motion_dir_list,flow_mag_list=get_flow_vector(bbox_flow_list)
        
        ##send the full OF of road #####
        road_OF=bbox_flow_list[-1]
        # of_2_world_frame.flow_3d(road_OF,road_mask_rectangle_bbox,count)

        ####### All objects OF results #######
        # print("Unit OpticalFlow Vector: ",motion_dir_list[:-1], " |  OpticalFlow magnitude: ",flow_mag_list[:-1])
        #### ROAD's OF result #####
        # print("ROAD-Unit OpticalFlow Vector: ",motion_dir_list[-1], " | ROAD-OpticalFlow magnitude: ",flow_mag_list[-1])


        ######### VIZ #########
        imggray = cv2.rectangle(imggray,(road_mask_rectangle_bbox[0][0],road_mask_rectangle_bbox[0][1]),(road_mask_rectangle_bbox[0][2],road_mask_rectangle_bbox[0][3]),(0,0,255),thickness=3)
        cv2.imshow('Flow in Bounding Boxes',draw_bbox_opticalflow(imggray,denseflow_fullimg,bboxes))
        # cv2.imshow('Flow in Bounding Boxes',draw_opticalflow_fullimg(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),denseflow_fullimg,step=16))
        # cv2.waitKey(0)   #incase you wanna pause and analyze 1 image
        if cv2.waitKey(100)==ord('q'):
            break
        # img = cv2.rectangle(img,(road_mask_rectangle_bbox[0][0],road_mask_rectangle_bbox[0][1]),(road_mask_rectangle_bbox[0][2],road_mask_rectangle_bbox[0][3]),(0,0,255),thickness=3)
        # cv2.imshow("road",img)
        # cv2.waitKey(0)
        count+=1
    
    cv2.destroyAllWindows()

    # print("time = ",sum(fps)/len(fps))
    #get moving /static objs based on class id
    #then get their bbox xyxy only.
    #do the optical flow only in this bboxes


if __name__=="__main__":
    main()