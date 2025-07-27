import threading
import queue
from typing import TYPE_CHECKING, Dict, Tuple
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial
import psutil
import math
import cv2
import numpy
import torch

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from app.processors.workers.frame_worker import FrameWorker
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions

from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class VideoProcessor(QObject):
    frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    single_frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    def __init__(self, main_window: 'MainWindow', num_threads=2):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue(maxsize=num_threads)
        self.media_capture: cv2.VideoCapture|None = None
        self.file_type = None
        self.fps = 0
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads
        self.threads: Dict[int, threading.Thread] = {}

        self.current_frame: numpy.ndarray = []
        self.recording = False

        self.recording_sp: subprocess.Popen|None = None 
        self.temp_file = '' 
        
        #Used to calculate the total processing time
        self.start_time = 0.0
        self.end_time = 0.0

        #Used to store the video start and enc seek time
        self.play_start_time = 0.0
        self.play_end_time = 0.0

        # Timer to manage frame reading intervals
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.process_next_frame)

        # Source video color characteristics (default to SDR)
        self.source_color_trc = 'bt709'
        self.source_color_primaries = 'bt709'
        self.source_color_space = 'bt709'
        self.source_pix_fmt = 'yuv420p' # Default SDR 8-bit
        self.source_bit_depth = 8       # Default SDR 8-bit

        self.next_frame_to_display = 0
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.frame_display_timer = QTimer()
        self.frame_display_timer.timeout.connect(self.display_next_frame)
        self.frames_to_display: Dict[int, Tuple[QPixmap, numpy.ndarray]] = {}

        # Timer to update the gpu memory usage progressbar 
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(partial(common_widget_actions.update_gpu_memory_progressbar, main_window))

        self.single_frame_processed_signal.connect(self.display_current_frame)

    Slot(int, QPixmap, numpy.ndarray)

    def store_frame_to_display(self, frame_number, pixmap, frame):
        self.frames_to_display[frame_number] = (pixmap, frame)

    # Use a queue to store the webcam frames, since the order of frames is not that important (Unless there are too many threads)
    Slot(QPixmap, numpy.ndarray)

    def display_current_frame(self, frame_number, pixmap, frame):
        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number, reset_fit=True)
            self.main_window.loading_new_media = False

        else:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number,)
        self.current_frame = frame
        torch.cuda.empty_cache()
        #Set GPU Memory Progressbar
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)

    def display_next_frame(self):
        if not self.processing or (self.next_frame_to_display > self.max_frame_number):
            self.stop_processing()
        if self.next_frame_to_display not in self.frames_to_display:
            return
        else:
            pixmap, frame = self.frames_to_display.pop(self.next_frame_to_display)
            self.current_frame = frame

            if self.recording:
                self.recording_sp.stdin.write(frame.tobytes())
            # Update the widget values using parameters if it is not recording (The updation of actual parameters is already done inside the FrameWorker, this step is to make the changes appear in the widgets)
            if not self.recording:
                video_control_actions.update_widget_values_from_markers(self.main_window, self.next_frame_to_display)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)
            try:
                self.threads.pop(self.next_frame_to_display)
            except:
                print("Thread not found in dict!")
            self.next_frame_to_display += 1

    def set_number_of_threads(self, value):
        self.stop_processing()
        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        print(f"Max Threads set as {value} ")

    def process_video(self):
        """Start video processing by reading frames and enqueueing them."""
        if self.processing:
            print("Processing already in progress. Ignoring start request.")
            return
            
        # Re-initialize the timers
        self.frame_display_timer = QTimer()
        self.frame_read_timer = QTimer()

        if self.file_type == 'video':
            self.frame_display_timer.timeout.connect(self.display_next_frame)
            self.frame_read_timer.timeout.connect(self.process_next_frame)

            if self.media_capture and self.media_capture.isOpened():
                print("Starting video processing.")
                if self.recording:
                    layout_actions.disable_all_parameters_and_control_widget(self.main_window)

                self.start_time = time.perf_counter()
                self.processing = True
                self.frames_to_display.clear()
                self.threads.clear()

                if self.recording:
                    self.create_ffmpeg_subprocess()

                self.play_start_time = float(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))

                if self.main_window.control['VideoPlaybackCustomFpsToggle']:
                    fps = self.main_window.control['VideoPlaybackCustomFpsSlider']
                else:
                    fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                
                interval = 1000 / fps if fps > 0 else 30
                interval = int(interval * 0.8) #Process 20% faster to offset the frame loading & processing time so the video will be played close to the original fps
                print(f"Starting frame_read_timer with an interval of {interval} ms.")
                if self.recording:
                    self.frame_read_timer.start()
                    self.frame_display_timer.start()
                else:
                    self.frame_read_timer.start(interval)
                    self.frame_display_timer.start()
                self.gpu_memory_update_timer.start(5000) #Update GPU memory progressbar every 5 Seconds

            else:
                print("Error: Unable to open the video.")
                self.processing = False
                self.frame_read_timer.stop()
                video_control_actions.set_play_button_icon_to_play(self.main_window)

    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""

        if self.current_frame_number > self.max_frame_number:
            # print("Stopping frame_read_timer as all frames have been read!")
            self.frame_read_timer.stop()
            return

        if self.frame_queue.qsize() >= self.num_threads:
            # print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return

        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = not self.recording)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame)
                self.current_frame_number += 1
            else:
               # print("Cannot read frame!", self.current_frame_number)
                self.stop_processing()
               # self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.\n Stopped Processing...!', self.main_window)

    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Start a FrameWorker to process the given frame."""
        worker = FrameWorker(frame, self.main_window, frame_number, self.frame_queue, is_single_frame)
        self.threads[frame_number] = worker
        if is_single_frame:
            worker.run()
        else:
            worker.start()

    def process_current_frame(self):
        self.next_frame_to_display = self.current_frame_number
        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode=False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
                
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print("Cannot read frame!", self.current_frame_number)
               # self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.', self.main_window)

        # """Process a single image frame directly without queuing."""
        elif self.file_type == 'image':
            frame = misc_helpers.read_image_file(self.media_path)
            if frame is not None:

                frame = frame[..., ::-1]  # Convert BGR to RGB
                self.frame_queue.put(self.current_frame_number)
                # print("Processing current frame as image.")
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Error: Unable to read image file.")

        self.join_and_clear_threads()

    # @misc_helpers.benchmark
    def stop_processing(self):
        """Stop video processing and signal completion."""
        if not self.processing:
            # print("Processing not active. No action to perform.")
            video_control_actions.reset_media_buttons(self.main_window)

            return False
        
        print("Stopping video processing.")
        self.processing = False
        
        if self.file_type=='video':

            # print("Stopping Timers")
            self.frame_read_timer.stop()
            self.frame_display_timer.stop()
            self.gpu_memory_update_timer.stop()
            self.join_and_clear_threads()


            # print("Clearing Threads and Queues")
            self.threads.clear()
            self.frames_to_display.clear()

            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()

            self.current_frame_number = self.main_window.videoSeekSlider.value()
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)

            if self.recording and self.file_type=='video':
                self.recording_sp.stdin.close()
                self.recording_sp.wait()

            self.play_end_time = float(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))

            if self.file_type=='video':
                if self.recording:
                    final_file_path = misc_helpers.get_output_file_path(self.media_path, self.main_window.control['OutputMediaFolder'])
                    if Path(final_file_path).is_file():
                        os.remove(final_file_path)
                    print("Adding audio...")
                    args = ["ffmpeg",
                            '-hide_banner',
                            '-loglevel',    'error',
                            "-i", self.temp_file,
                            "-ss", str(self.play_start_time), "-to", str(self.play_end_time), "-i",  self.media_path,
                            "-c",  "copy", # may be c:v
                            "-map", "0:v:0", "-map", "1:a:0?",
                            "-shortest",
                            final_file_path]
                    subprocess.run(args, check=False) #Add Audio
                    os.remove(self.temp_file)

                self.end_time = time.perf_counter()
                processing_time = self.end_time - self.start_time
                print(f"\nProcessing completed in {processing_time} seconds")
                avg_fps = ((self.play_end_time - self.play_start_time) * self.fps) / processing_time
                print(f'Average FPS: {avg_fps}\n')

                if self.recording:
                    layout_actions.enable_all_parameters_and_control_widget(self.main_window)

            self.recording = False #Set recording as False to make sure the next process_video() call doesnt not record the video, unless the user press the record button

            print("Clearing Cache")
            torch.cuda.empty_cache()
            gc.collect()
            video_control_actions.reset_media_buttons(self.main_window)
            print("Successfully Stopped Processing")
            return True
        
    def join_and_clear_threads(self):
        # print("Joining Threads")
        for _, thread in self.threads.items():
            if thread.is_alive():
                thread.join()
        # print('Clearing Threads')
        self.threads.clear()
    
    def create_ffmpeg_subprocess(self):
        # Use Dimensions of the last processed frame as it could be different from the original frame due to restorers and frame enhancers 
        control = self.main_window.control.copy()
        enhancer_type = control['FrameEnhancerTypeSelection']
        frame_height, frame_width, _ = self.current_frame.shape
        frame_height_down = frame_height
        frame_width_down = frame_width
        frame_height_down_mult = frame_height / 1080
        frame_width_down_mult = frame_width / 1920
#        if control['FrameEnhancerEnableToggle'] and control['FrameEnhancerDownToggle']:
#            if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
#                frame_height_down = math.ceil(frame_height / 2)
#                frame_width_down = math.ceil(frame_width / 2)
#            else:
#                frame_height_down = math.ceil(frame_height /4)
#                frame_width_down = math.ceil(frame_width /4)
                
        if control['FrameEnhancerDownToggle']:
            if frame_width / frame_height == 16 / 9:
                frame_height_down = math.ceil(frame_height / frame_height_down_mult)
                frame_width_down = math.ceil(frame_width / frame_width_down_mult)
            else:
                Print('Not 16/9 ratio')

        self.temp_file = r'temp_output.mp4'
        if Path(self.temp_file).is_file():
            os.remove(self.temp_file)

        #args = [
        #    "ffmpeg",
        #    "-hide_banner",
        #    "-loglevel", "error",
        #    "-f", "rawvideo",             # Specify raw video input
        #    "-pix_fmt", "bgr24",          # Pixel format of input frames
        #    "-s", f"{frame_width}x{frame_height}",  # Frame resolution
        #    "-r", str(self.fps),          # Frame rate
        #    "-i", "pipe:",                # Input from stdin
        #    "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",  # Padding and format conversion            
        #    "-c:v", "libx264",            # H.264 codec
        #    "-crf", "18",                 # Quality setting
        #    self.temp_file                # Output file
        #]
        # Consistently output SDR BT.709 as the Python pipeline produces 8-bit SDR frames.
        output_pix_fmt = 'yuv420p' # Standard 8-bit SDR
        output_color_trc = 'bt709'
        output_colorspace = 'bt709'
        output_color_primaries = 'bt709'

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",

            # Input 1: Raw video frames from Python via pipe (CPU memory)
            "-f", "rawvideo",
            "-pix_fmt", "bgr24", # Pixel format of the data being piped in
            "-s", f"{frame_width}x{frame_height}", # Resolution of the piped data
            "-r", str(self.fps),
            "-i", "pipe:",

            # Video Filtergraph
            # The bgr24 input from pipe will be converted by ffmpeg.
            # For NVENC, it needs a GPU compatible format like nv12 (8-bit) or p010le (10-bit).
            # The `format` filter handles CPU-side conversion before hwupload.
        ]
        # Since input to pipe is 8-bit bgr24, always use nv12 for NVENC.
        # The filter chain should be: bgr24 (input from pipe) -> format=nv12 (CPU conversion) -> hwupload_cuda -> hevc_nvenc
        if self.main_window.control.get('ProvidersPrioritySelection') in ["CUDA", "TensorRT", "TensorRT-Engine"]:
            # For NVENC, frames must be in GPU memory.
            # Let hevc_nvenc handle the upload and format conversion by specifying input format.
            # No explicit -vf for hwupload/format needed if nvenc handles it.
            pass # hevc_nvenc will use the -pix_fmt bgr24 from input and manage upload/conversion
        else: # CPU processing for ffmpeg (e.g. libx264)
            args.extend(["-vf", f"format={output_pix_fmt}"])

        args.extend([
            # Video Codec: Use NVIDIA HEVC encoder
            "-c:v", "hevc_nvenc",
            # Quality Setting: NVENC uses -cq (Constant Quality scale, lower=better, ~18-28 is common range)
            # Or use -qp (Constant Quantization Parameter)
            # Or target bitrate: -b:v 60M
            "-cq", "18", # Experiment with this value
            # Preset: Controls speed vs quality trade-off for NVENC (e.g., p1-p7, default is p4/p5)
            # p5=medium, p6=slow, p7=slower (higher quality)
            "-preset", "p6",
            # Profile/Level might still be useful
            # "-profile:v", "main",
            # "-level:v", "6.2",
            # Set color properties (NVENC should respect these)
            "-vf", f"scale={frame_width_down}x{frame_height_down}:flags=lanczos",
            "-pix_fmt", output_pix_fmt, # Explicitly set output pixel format for the container
            "-color_range", "tv",
            "-colorspace", output_colorspace,
            "-color_primaries", output_color_primaries,
            "-color_trc", output_color_trc,
            "-tag:v", "hvc1",

            # Audio Codec: Copy directly
            "-c:a", "copy",

            # Output File
            self.temp_file
            
        ])

        self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE)
