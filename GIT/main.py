import cv2
import numpy as np
import threading
import time
from tensorflow.lite.python.interpreter import Interpreter

from class_lane_keeping import LaneDetectionThread
from class_object_detection import ObjectDetector

if __name__ == "__main__":
    cap1 = cv2.VideoCapture("D:\\OKhe\\bosch\\test_multithreading\\Thu4\\test_map4.mp4")#D:\\OKhe\\bosch\\test_multithreading\\Thu4\\test_map4.mp4

    while True:
        # Create instances of MyThread1 and CameraCaptureThread
        thread_object = ObjectDetector(cap1)
        thread_lane = LaneDetectionThread(cap1)


        # Start both threads
        thread_object.start()
        thread_lane.start()

        # Wait for both threads to finish
        thread_object.join()
        thread_lane.join()
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    print("All threads have finished")
