import cv2
import numpy as np
import argparse

confid = 0.5
thresh = 0.5
mouse_pts = []                                                   

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        


def calculate_social_distancing(vid_path):
    vs = cv2.VideoCapture(vid_path)    
        
    global image
    
    (_, frame) = vs.read()
        
    
    while True:
        image = frame
        cv2.imshow("image", image)
        cv2.waitKey(1)
        if len(mouse_pts) == 4:
            print(mouse_pts)
            cv2.destroyWindow("image")
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example.mp4' ,
                    help='Path for input video')
                    
                    
    values = parser.parse_args()

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)
    
    calculate_social_distancing(values.video_path)



