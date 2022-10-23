import cv2
import numpy as np
import os.path as osp
import time
import glob
import os
import constants
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
import multiprocessing
from util.aws_utils import sqs_transfer
from util.aws_utils import s3_transfer

class DBManager:
	def __init__(self, source, width, height, fps, output_dir):
		# self.video_dir = Path(Path(output_dir) / 'VIDEO')
		# self.video_dir.mkdir(parents=True, exist_ok=True)
		self.image_dir = Path(Path(output_dir) / 'RECORD')
		self.image_dir.mkdir(parents=True, exist_ok=True)

		self.filename=source.split('/')[-1].split('.')[0]
		self.width = width
		self.height = height
		self.fps = fps
		self.record = None
		self.record_frame_cnt = 0

	def snap_image(self, frame, viol_text):
		now = datetime.now()
		dt_string = now.strftime("%Y%m%d%H%M%S")
		s_img_name = osp.splitext(osp.basename(self.filename))[0] + "_" + viol_text + dt_string + ".png"
		snap_path = self.image_dir
		snap_imgname = osp.join(snap_path, s_img_name)
		print(f'[snap_image]  snap violation image  {snap_imgname}')
		cv2.imwrite(snap_imgname, frame)
		return snap_path, s_img_name

	def record_start(self, frame):
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		now = datetime.now()
		dt_string = now.strftime("%Y%m%d%H%M%S")

		if self.record == None:
			self.record = cv2.VideoWriter(f"{self.video_dir}/{self.filename}_{dt_string}.avi", fourcc, self.fps, (int(self.width*constants.OUTPUT_RES_RATIO), int(self.height*constants.OUTPUT_RES_RATIO)))

		self.record_frame_cnt = 5*self.fps - 1
		self.record.write(frame)
		self.record_frame_cnt -= 1

	def record_update(self, frame):
		if self.record == None:
			return

		if self.record_frame_cnt == 0:
			self.record = None
			return
		self.record.write(frame)
		self.record_frame_cnt -= 1

s3_uploader = s3_transfer()
sqs_push = sqs_transfer()

annot_frame_buffer = deque(maxlen=600)
frames_id_buffer = deque(maxlen=600)

def handle_annot_frames_buffer(frame,frame_id):
    annot_frame_buffer.appendleft(frame)
    frames_id_buffer.appendleft((frame_id%2000))

def convert_video_h264(local_file,video_duration_in_s,auto_delete=True):
    new_file = (osp.splitext(local_file)[0]+'_.mp4')
    cmd = "ffmpeg -i "+ local_file +" -vcodec libx264 " + new_file
    os.system(cmd)
    time.sleep(video_duration_in_s)
    os.remove(local_file)
    os.rename(new_file,local_file)

def process_violation_video(local_video_fname,s3_filename,frame_id,viol_txt,violation_id,width,height,video_duration_in_s=5,fps=12):
    # Check video recording duration (current limit 30 s)
    # logger.info(str(list(frames_id_buffer)))
    video_duration_in_s = video_duration_in_s if (video_duration_in_s <= 30 and video_duration_in_s > 0) else 10
    # Make the thread on sleep first 
    # time.sleep(video_duration_in_s)
    # Video writer init
    vid_writer = cv2.VideoWriter(local_video_fname, cv2.VideoWriter_fourcc(*"mp4v"), int(fps),(int(width), int(height)))
    # Keep on checking the video buffers
    while list(frames_id_buffer).index((frame_id%2000)) < (video_duration_in_s*fps):
        # logger.info(str(list(frames_id_buffer).index((frame_id%2000))))
        continue
    frames_id_list_copy = list(frames_id_buffer)
    frames_list_copy = list(annot_frame_buffer)
    first_frame_ind = max(0,int((frames_id_list_copy.index((frame_id%2000))) - (video_duration_in_s*fps)))
    # logger.debug(str(first_frame_ind))
    frames_list_copy = frames_list_copy[first_frame_ind:frames_id_list_copy.index((frame_id%2000))]
    for each_frame in reversed(frames_list_copy):
        # print(each_frame.shape)
        vid_writer.write(each_frame)
    vid_writer.release()
    # Add module to process video to h264 codec
    convert_video_h264(local_file=local_video_fname,video_duration_in_s=video_duration_in_s)
    _,obj_url = s3_uploader.s3_file_transfer(local_file=local_video_fname,s3_file=s3_filename,violation_id=violation_id)
    sqs_status = sqs_push.push_msg(kinesis_name=os.environ.get('kinesis_url'),msg_type='video',violation_id=violation_id,category='PPE',subcategory=viol_txt,object_url=obj_url)    

def s3_sqs_handler(local_filepath, local_filename,s3_filename,viol_txt,frame_id,height,width,fps,process_video = True):
    # logger.debug("Violation of s3 + sqs going to be processed as  " + str(viol_txt))
    violation_id = 0
    filename = osp.join(local_filepath, local_filename)
    violation_id,obj_url = s3_uploader.s3_file_transfer(local_file=filename,s3_file=s3_filename)
    sqs_status = sqs_push.push_msg(msg_type='image',violation_id=violation_id,kinesis_name=os.environ.get('kinesis_url'),category='PPE',subcategory=viol_txt,object_url=obj_url)
    video_duration_in_s = 5.0
    if process_video:
        s3_video_thread = threading.Timer(video_duration_in_s*1.5,process_violation_video,args=(osp.join(local_filepath, (osp.splitext(local_filename)[0]+'.mp4')),(osp.splitext(s3_filename)[0]+'.mp4'),frame_id,viol_txt,violation_id,width,height,video_duration_in_s,fps))
        # s3_video_thread.daemon = True
        s3_video_thread.start()
        # process_violation_video(local_video_fname=(osp.splitext(local_filename)[0]+'.mp4'),s3_filename=(osp.splitext(s3_filename)[0]+'.mp4'),frame_id=frame_id,viol_txt=viol_txt,violation_id=violation_id,width=width,height=height,video_duration_in_s=5)