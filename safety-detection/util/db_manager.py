import cv2
import numpy as np
import os.path as osp
import time
import glob
import os
import constants
from pathlib import Path
from datetime import datetime

class DBManager:
	def __init__(self, source, width, height, fps, output_dir):
		self.video_dir = Path(Path(output_dir) / 'VIDEO')
		self.video_dir.mkdir(parents=True, exist_ok=True)
		self.image_dir = Path(Path(output_dir) / 'IMAGE')
		self.image_dir.mkdir(parents=True, exist_ok=True)

		self.filename=source.split('/')[-1].split('.')[0]
		self.width = width
		self.height = height
		self.fps = fps
		self.record = None
		self.record_frame_cnt = 0

	def snap_image(self, frame):
		now = datetime.now()
		dt_string = now.strftime("%Y%m%d%H%M%S")
		s_img_name = osp.splitext(osp.basename(self.filename))[0] + "_" + dt_string + ".png"
		snap_path = self.image_dir
		snap_imgname = osp.join(snap_path, s_img_name)
		print(f'[snap_image]  snap violation image  {snap_imgname}')
		cv2.imwrite(snap_imgname, frame)

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


