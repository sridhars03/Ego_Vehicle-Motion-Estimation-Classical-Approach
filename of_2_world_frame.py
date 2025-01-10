import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from road_segmentation import road_segment

##### Camera Projection Matrix for NuScenes Images dataset ######

#### Intrinsics ####
##### CAM_FRONT ####
K_CAM_FRONT=np.array([[1266.417203046554,0.0,816.2670197447984],
            [0.0,1266.417203046554,491.50706579294757],
            [0,0,1]])

#### Extrinsics ####
##### CAM_FRONT ####
R_CAM_FRONT_quaternion=np.array([0.4998015430569128,-0.5030316162024876,0.4997798114386805,-0.49737083824542755])
R_CAM_FRONT=R.from_quat(R_CAM_FRONT_quaternion).as_matrix()

t=np.array([[1.70079118954], [0.0159456324149], [1.51095763913]])

extrinsic_CAM_FRONT=np.hstack((R_CAM_FRONT,t))

P_CAM_FRONT= K_CAM_FRONT @ extrinsic_CAM_FRONT

#### End of Camera Projection Matrix for NuScenes Images dataset #######


##### Convert pixel to world coord #####
# depth =5            ###to be found
# pixel_coord=np.array([100,100])

# cam_coords_FRONT=(np.linalg.inv(K_CAM_FRONT) @ pixel_coord) * depth
# world_coords_FRONT=R.T @ (cam_coords_FRONT - t)




#######################################################
# Function to compute 3D flow in world coordinates
def compute_3d_flow(optical_flow,depth_map,K,R,t):
    """
    Computes 3D flow in world coordinates from optical flow, depth map, and camera parameters.

    Args:
        optical_flow: Optical flow array of shape (H, W, 2) -> displacement (u, v).
        depth_map: Depth map of shape (H, W) -> depth values Z for each pixel.
        K: Intrinsic matrix of the camera (3x3).
        R: Rotation matrix (3x3) from camera to world.
        t: Translation vector (3x1) from camera to world.

    Returns:
        flow_3d_world: 3D flow vectors in world coordinates (H, W, 3).
    """
    # Get image dimensions
    H,W=depth_map.shape
    
    # Create pixel coordinates (u, v)
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    pixel_coords = np.stack([u_coords, v_coords, np.ones_like(u_coords)], axis=-1)  # Shape: (H, W, 3)

    # Reshape for easier computation
    pixel_coords = pixel_coords.reshape(-1, 3)  # Shape: (N, 3)
    optical_flow = optical_flow.reshape(-1, 2)  # Shape: (N, 2)
    depth_map = depth_map.flatten()            # Shape: (N,)

    K_inv = np.linalg.inv(K)
    points_3d_camera = depth_map[:, None] * (K_inv @ pixel_coords.T).T  # Shape: (N, 3)

    displaced_pixel_coords = pixel_coords[:, :2] + optical_flow  # Shape: (N, 2)
    displaced_pixel_coords_hom = np.hstack([displaced_pixel_coords, np.ones((len(displaced_pixel_coords), 1))])
    displaced_points_3d_camera = depth_map[:, None] * (K_inv @ displaced_pixel_coords_hom.T).T  # Shape: (N, 3)

    flow_3d_camera = displaced_points_3d_camera - points_3d_camera  # Shape: (N, 3)
    flow_3d_world = (R @ flow_3d_camera.T).T

    #Reshape back to image dimensions
    flow_3d_world = flow_3d_world.reshape(H, W, 3)

    return flow_3d_world


depth_file_path="G://AV projects//CV_project//combined_depth_arrays.npy"
depth=np.load(depth_file_path)
target_size=(depth.shape[1]//2,depth.shape[2]//2)  # (Height, Width)

downsampled_depth=np.array([resize(img,target_size,mode='reflect',anti_aliasing=True) for img in depth])
# print("Resized Depth array Shape: ",downsampled_depth.shape)
downsampled_depth=10*(1-downsampled_depth)
# print(downsampled_depth[1],downsampled_depth[10])

def flow_3d(optical_flow,bbox,count,K_CAM_FRONT=K_CAM_FRONT,R_CAM_FRONT=R_CAM_FRONT,t=t):
    
    depth_map=downsampled_depth[count]
    depth_map=depth_map[int(bbox[0][1]):int(bbox[0][3]), int(bbox[0][0]):int(bbox[0][2])]
    flow_3d_world=compute_3d_flow(optical_flow,depth_map,K_CAM_FRONT,R_CAM_FRONT,t)
    
    #####  maybe do some viz ######
    # print("********WORKS 3d flow shape= ",flow_3d_world.shape)

    #### X and Z motion = right and forward dir of cam ####
    road_flow_3d_X=flow_3d_world[...,0]
    road_flow_3d_Z=flow_3d_world[...,2]
    road_flow_3d_Y=flow_3d_world[...,1]

    car_motion_X=(-1)*road_flow_3d_X
    car_motion_Z=(-1)*road_flow_3d_Z
    car_motion_Y=(-1)*road_flow_3d_Y

    #### get the mean #####
    car_motion_Xmean=np.mean(car_motion_X)
    car_motion_Zmean=np.mean(car_motion_Z)
    car_motion_Ymean=np.mean(car_motion_Y)
    # print(car_motion_Xmean,car_motion_Ymean,car_motion_Zmean)

    # ########### VIZ Net Ego-Motion Vector ###########
    _, ax = plt.subplots(figsize=(8, 8))
    origin = [0, 0]
    net_magnitude = np.sqrt(car_motion_Xmean**2 + car_motion_Zmean**2)
    net_angle = np.arctan2(car_motion_Zmean, car_motion_Xmean) * (180 / np.pi)

    ax.quiver(
        origin[0], origin[1],  # Arrow starting point
        car_motion_Xmean, car_motion_Zmean,  # Arrow direction and magnitude
        angles='xy', scale_units='xy', scale=1, color='blue', linewidth=2, label="Net Motion Vector"
    )

    ax.set_xlabel("X-axis (Right Direction)", fontsize=10)
    ax.set_ylabel("Z-axis (Forward Direction)", fontsize=10)
    ax.set_title("Net Motion Vector of Car in X-Z Plane", fontsize=10)

    max_val = max(abs(car_motion_Xmean), abs(car_motion_Zmean)) * 1.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    ax.grid(True)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_aspect('equal')

    ax.text(
        car_motion_Xmean / 2, car_motion_Zmean / 2,  # Position of text (midpoint of arrow)
        f"Mag: {net_magnitude:.2f}\nAngle: {net_angle:.2f}°",
        fontsize=10, color='red', ha='center'
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
    # # Define example inputs
    # H, W = 480, 640  # Image dimensions
    
    # optical_flow = np.random.randn(H, W, 2) * 5  # Simulated optical flow
    # depth_map = np.ones((H, W)) * 10  # Assume constant depth of 10 meters for simplicity

    # #Calc 3D flow in world coordinates
    # flow_3d(optical_flow,depth_map,K_CAM_FRONT,R_CAM_FRONT,t)