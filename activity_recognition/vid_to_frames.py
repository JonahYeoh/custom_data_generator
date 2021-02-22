import os
import numpy as np
import cv2

def vid_to_frames(root_dir, dest_dir, rescale=True, preserved_aspect_ratio=True, scale=0.8):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    cat_list = os.listdir(root_dir) # train, test, val
    for cat in cat_list:
        cat_path = os.path.join(root_dir, cat)
        activity_list = os.listdir(cat_path)
        cat_dest_path = os.path.join(dest_dir, cat)
        if not os.path.exists(cat_dest_path):
            os.mkdir(cat_dest_path)
        for activity in activity_list: # loop over every activity folder
            activity_path = os.path.join(cat_path,activity) # 'UCF-101/Archery'
            dest_activity_path = os.path.join(cat_dest_path,activity) # 'activity_data/Archery'
            if not os.path.exists(dest_activity_path):
                os.mkdir(dest_activity_path)
            write_frames(activity_path,dest_activity_path, rescale, preserved_aspect_ratio, scale)

def write_frames(activity_path,dest_activity_path, rescale, preserved_aspect_ratio, scale, frame_format='jpg', abs_dim=None):
    # read the list of video from 'UCF-101/train/Archery' - [v_Archery_g01_c01.avi,v_Archery_g01_c01.avi, ......]
    vid_list = os.listdir(activity_path)
    print(vid_list)
    for vid in vid_list: # v_Archery_g01_c01.avi
        dest_folder_name = vid[:-4] # v_Archery_g01_c01
        dest_folder_path = os.path.join(dest_activity_path,dest_folder_name) # 'activity_data/train/Archery/v_Archery_g01_c01'
        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)
            
        vid_path = os.path.join(activity_path,vid)  # 'UCF-101/train/Archery/v_Archery_g01_c01.avi'
        print ('video path: ', vid_path)
        cap = cv2.VideoCapture(vid_path) # initialize a cap object for reading the video
        
        ret=True
        frame_num=0
        while ret:
            ret, img = cap.read()
            output_file_name = 'img_{:06d}'.format(frame_num) + '.{}'.format(frame_format) # img_000001.png
            # output frame to write 'activity_data/Archery/v_Archery_g01_c01/img_000001.png'
            output_file_path = os.path.join(dest_folder_path, output_file_name)
            frame_num += 1
            print("Frame no. ", frame_num)
            try:
                # cv2.imshow('img',img)
                # cv2.waitKey(5)
                # downscale, preserved aspect ratio
                if rescale:
                    if preserved_aspect_ratio:
                        dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    else:
                        dim = abs_dim
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(output_file_path, img) # writing frames to defined location
            except Exception as e:
                print(e)
            if ret==False:
                cv2.destroyAllWindows()
                cap.release()

if __name__ == '__main__':
    root = 'G:\\video_data\\UCF50_split\\'
    dest = 'G:\\video_data\\activity_file\\data_files\\'
    vid_to_frames(root, dest)
    
