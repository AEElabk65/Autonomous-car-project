
import numpy as np
import cv2
import time
import threading
from tensorflow.lite.python.interpreter import Interpreter
from CountsPerSec import CountsPerSec

class ObjectDetector(threading.Thread):
    def __init__(self, cap):
        super().__init__()  # Gọi hàm khởi tạo của lớp cha
        # Khởi tạo đối tượng detector
        self.model_path = "model_path"
        self.label_path = "label_path"
        self.min_confidence = 0.5
        self.cps = CountsPerSec().start()
        
        self.cap = cap
        
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.current_command = "none"
        self.float_input = (self.input_details[0]['dtype'] == np.float32)
        
        with open(self.label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
    def run(self):
        cv2.startWindowThread()
        while True:
            ret, frame = self.cap.read()
            #self.cps.increment() 
            if not ret:
                print("Failed to capture frame")
                break

            frame_output,signal2=self.detect_objects(frame) 
            if signal2 =='0.0':
                self.send_command_signal(1,0)

            cps_frame = self.putIterationsPerSec(frame_output, self.cps.countsPerSec())  
            resized_image = cv2.resize(cps_frame, (700, 800))  
            cv2.imshow('output',resized_image)        
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break        
        self.cap.release()
        cv2.destroyAllWindows()   

    def send_command_signal(self, msgID, angle):
        if angle != self.current_command:
            self.current_command = angle
            print("Sending command:", (msgID, self.current_command))

    def putIterationsPerSec(self,frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """
        cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=3)

        return frame

        
    def detect_objects(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)
        
        if self.float_input:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        signal='none'
        for i in range(len(scores)):
            if scores[i] > self.min_confidence and scores[i] <= 1.0:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))

                object_name = self.labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                distance = int((-0.4412)*int(xmax - xmin)+63.9706)


                
                if int(classes[i])==0:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                    cv2.putText(frame, f'Name:{label}, Distance: {distance}', (xmin, label_ymin-7),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    signal=classes[i]
                if int(classes[i])==1:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==2:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 255, 0), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==3:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 255), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==4:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 235), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==6:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 245), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==7:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 225), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==8:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 215), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                if int(classes[i])==9:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 0, 205), 2)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    signal=classes[i]
                
            return(frame,signal)
                

# Mở video capture
cap =  cv2.VideoCapture("D:\\OKhe\\bosch\\Traffic_sign\\5292766527050.mp4") #"D:\\OKhe\\bosch\\test_multithreading\\Thu4\\test_map3.mp4" #D:\\OKhe\\bosch\\Traffic_sign\\object.mp4
# Tạo một đối tượng ObjectDetector
detector = ObjectDetector(cap)
# Bắt đầu luồng
detector.start()
