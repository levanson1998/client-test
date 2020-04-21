#-------------Dai's Libs------------#
import sys
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IEPlugin
# import skfuzzy as fuzz
# from skfuzzy import control as ctrl
import heapq
import threading
from imutils.video.pivideostream import PiVideoStream
#from imutils.video.filevideostream import FileVideoStream
import imutils

#-------------Tien's Libs-----------#
import subprocess
import os
import glob
import RPi.GPIO as GPIO
import datetime
import spidev
import binascii
import TimerClass
import socketio
import struct
from threading import Timer
import base64
import urllibs

#------------Region for global variables-----------------
#Tracking variables
# init_fuzzy_check = 0
lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
camera_width  = 320
camera_height = 240
window_name = ""
elapsedtime = 0.0
vs = None
number_of_ncs = 1
vidfps = 30
LABELS = [['background', 'logo_BK']]
right_pwm = None
left_pwm = None
right_pwm_output = 0
left_pwm_output = 0

isTracking = 0
ePosition = 0
eDistance = 0
frame = None
isTrackingBuffer = 0
ePositionBuffer = 0
eDistanceBuffer = 0
frameBufferToServer = None
spi = None

#Server Variables
#Constant
timerLost = 0
sio = socketio.Client()
sioIsConnected=False

#camera=PiCamera()
buzzer_pin = 12     #Pin config
tmr1 = None
tmr100 = None
#------------Functions for Logo Tracking and VPU Connection-----------------
def camThread(LABELS, results, frameBuffer, vidfps, video_file_path, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer):
    # global init_check   #This variable is used to init fuzzy logic controller at first initializing
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name
    global vs
    global camera_width
    global camera_height
    global right_pwm
    global left_pwm
    global isTracking
    global ePosition
    global eDistance

    if video_file_path != "":
        vs = FileVideoStream(video_file_path).start()
        window_name = "Movie File"
    else:
        vs = PiVideoStream((camera_width, camera_height), vidfps).start()
        window_name = "PiCamera"
    time.sleep(2)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        # PiCamera Stream Read
        color_image = vs.read()
        #color_image = cv2.flip(color_image, 1)
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        #Reinitialize  width and height of camera
        camera_height = color_image.shape[0]
        camera_width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        frameBufferToServer.put(color_image.copy())

        res = None

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, LABELS, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, LABELS, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer)

        cv2.imshow(window_name, cv2.resize(imdraw, (camera_width, camera_height)))

        if cv2.waitKey(1)&0xFF == ord('q'):
            sys.exit(0)
            

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Playback) {:.1f} FPS".format(time1/15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2 - t1
        time1 += 1/elapsedTime
        time2 += elapsedTime

def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):
    while True:
        ncsworker.predict_async()


class NcsWorker(object):
    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs):
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = "./mo_caffe/no_bn.xml"
        self.model_bin = "./mo_caffe/no_bn.bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device = "MYRIAD")
        self.net = IENetwork(model = self.model_xml, weights = self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network = self.net, num_requests = self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs

    def image_preprocessing(self, color_image):

        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg - 127.5
        prepimg = prepimg * 0.007843
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        # print("prepimg", prepimg)
        return prepimg

    def predict_async(self):
        try:
            if self.frameBuffer.empty():
                return

            prepimg = self.image_preprocessing(self.frameBuffer.get())
            reqnum = searchlist(self.inferred_request, 0)
            # print("reqnum", reqnum)

            if reqnum > -1:
                self.exec_net.start_async(request_id = reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            if (len(self.heap_request) >= 0):
                cnt, dev = heapq.heappop(self.heap_request)

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)
                out = self.exec_net.requests[dev].outputs["detection_out"].flatten()
                # print("out", out)
                self.results.put([out])
                self.inferred_request[dev] = 0  
            else:
                heapq.heappush(self.heap_request, (cnt, dev))

        except:
            import traceback
            traceback.print_exc()


def inferencer(results, frameBuffer, number_of_ncs):
    global camera_width
    global camera_height
    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target = async_infer, 
            args = (NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs),))
        thworker.start()
        threads.append(thworker)    
    for th in threads:
        th.join()

def overlay_on_image(frames, object_infos, LABELS, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer):
    #Init variables for fuzzy output
    global camera_width
    global camera_height
    global isTracking
    global ePosition
    global eDistance

    try:             
        color_image = frames
        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()

        for (object_info, LABEL) in zip(object_infos, LABELS):
            class_id_array = []
            # print("object_info", object_info)
            # print("LABEL", LABEL)
            drawing_initial_flag = True

            for box_index in range(100):
                if object_info[box_index + 1] == 0.0:
                    break
                base_index = box_index * 7
                if (not np.isfinite(object_info[base_index]) or
                    not np.isfinite(object_info[base_index + 1]) or
                    not np.isfinite(object_info[base_index + 2]) or
                    not np.isfinite(object_info[base_index + 3]) or
                    not np.isfinite(object_info[base_index + 4]) or
                    not np.isfinite(object_info[base_index + 5]) or
                    not np.isfinite(object_info[base_index + 6])):
                    continue

                x1 = max(0, int(object_info[base_index + 3] * height))
                y1 = max(0, int(object_info[base_index + 4] * width))
                x2 = min(height, int(object_info[base_index + 5] * height))
                y2 = min(width, int(object_info[base_index + 6] * width))

                object_info_overlay = object_info[base_index:base_index + 7]

                min_score_percent = 95

                source_image_width = width
                source_image_height = height

                base_index = 0
                class_id = object_info_overlay[base_index + 1]

                percentage = int(object_info_overlay[base_index + 2] * 100)
                if (percentage <= min_score_percent):
                    continue
                             
                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                label_text = LABEL[int(class_id)] + " (" + str(percentage) + "%)"

                class_id_array.append(class_id) #Class ID container is used to detect any logo_BK class detected                
                ePosition = camera_width/2 - (box_left + box_right)/2
                #eDistance = camera_height - abs(y2 - y1)
                eDistance = abs(box_top - box_bottom) - 100


                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
                # cv2.putText(img_cp, "Diem dau", (box_left, box_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # cv2.putText(img_cp, "Diem cuoi", (box_right, box_right), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            
            class_id_array = np.array([class_id_array])
            isTracking = 1 if (np.where(class_id_array == LABELS[0].index("logo_BK"))[0].shape[0] > 0) else 0
            ePosition = ePosition if (isTracking == 1) else -1
            eDistance = eDistance if (isTracking == 1) else -1

            # print("isTracking", isTracking)
            # print("ePosition", ePosition)
            # print("eDistance", eDistance)

            isTrackingBuffer.put(isTracking)
            ePositionBuffer.put(ePosition)
            eDistanceBuffer.put(eDistance)

        cv2.putText(img_cp, fps,       (width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        return img_cp

    except:
        import traceback
        traceback.print_exc()

#----------------------Functions for Server tasks---------------------
#-------------SOCKET IO-------------
@sio.on('connect')
def on_connect():
    print("I'm connected!")
    sio.emit('suitcase-on',True)		

@sio.on('suitcase-send-status-ok')
def on_message(data):
    print('Server has received your data')
    sioIsConnected=True

@sio.on('suitcase-send-img-ok')
def on_message(data):
    print('Server has received your image')

@sio.on('disconnect')
def on_disconnect():
    sio.connect("https://suitcase-server.herokuapp.com")
    print("I'm disconnected")


def connected():
    try:
        host='https://suitcase-server.herokuapp.com'
        urllib.urlopen(host)
        return True
    except:
        return False

#-------------FUNCTIONS-------------
def init():
    global buzzer_pin
    #GPIO for Buzzer
    GPIO.setwarnings(False) #disable warnings
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer_pin,GPIO.OUT)    

def SpiToArm(isTrack, eP, eD):
    
    #Convert Int to Byte
    eP = round(eP)
    eD = round(eD)
    global spi

    tmp_P = eP.to_bytes(2, byteorder = "little", signed = True)
    eP_L = tmp_P[0]
    eP_H  = tmp_P[1]
    tmp_D = eD.to_bytes(2, byteorder = "little", signed = True)
    eD_L = tmp_D[0]
    eD_H = tmp_D[1]
    #Send data 7 bytes
    resp = spi.xfer2([isTrack, eP_L, eP_H, eD_L, eD_H, 0x0D, 0x0A])
    
    
    print("sendD: ", eD)
    print("sendP: ", eP)
    '''
    sleep(0.05)
    #Read speed from ARM   
    c = spi.readbytes(8)
    sleep(0.02)
    print(c)
    #resp=[LSB1...MSB1 LSB2 MSB2], exactly what we want
    firstFloat_ByteList=c[0:4]
    secondFloat_ByteList=c[4:8]

    #Get Bytes from list
    firstFloat_Bytes=bytes(firstFloat_ByteList)
    secondFloat_Bytes=bytes(secondFloat_ByteList)

    #Unpack those value, return tuple type
    left_pwm = struct.unpack('f',firstFloat_Bytes)
    right_pwm = struct.unpack('f',secondFloat_Bytes)
    
    print(left_pwm)
    print(right_pwm)
    '''
def CaptureImage():
    global frame
    # camera.start_preview()
    # camera.capture('/home/pi/Desktop/Storage/image.jpg')
    # camera.stop_preview()
    # img = cv2.imread('/home/pi/Desktop/Storage/image.jpg')
    #img = cv2.resize(frame,(320,240))    
    #cv2.imwrite('/home/pi/saved_images/image.jpg', img)
    cv2.imwrite('/home/pi/saved_images/image.jpg', frame)
    print('Done capture')

def BuzzerWarning():
    global buzzer_pin
    GPIO.output(buzzer_pin, GPIO.HIGH)
    sleep(1)
    GPIO.output(buzzer_pin, GPIO.LOW)
    sleep(10)

#---------------TIMER TASKS---------------
def Task1s():
    global isTracking
    global timerLost
    global sio
    global sioIsConnected
    sioIsConnected=False
    print('-------------------------------------')
    print('SENDING TO SERVER: ')
    if (connected==False):
        print("NO CONNECTION AVAIABLE!!!!!!!!")
        
    mydict = {"isTracking": isTracking, "lostTime": timerLost}
    sio.emit('suitcase-send-status', mydict)
    
    print(mydict)
    
    if (isTracking == 0):
        timerLost += 1
        CaptureImage()
        with open("/home/pi/saved_images/image.jpg","rb") as file:
            jpg_as_text = base64.b64encode(file.read())
        capturedTime = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        mydict_img = {"Image": "data:image/jpg;base64," + jpg_as_text.decode("utf-8"), "CapTime":capturedTime}
        sio.emit('suitcase-send-img', mydict_img)
        
        
    
    print('DONE UPLOADING TO SERVER-------------', isTracking)
    if (sioIsConnected==False):
        sio.connect("https://suitcase-server.herokuapp.com")

        
def Task100ms():
    print('--------------------------------------------------')
    global isTracking
    global timerLost
    global ePosition
    global eDistance
    global buzzer_pin
    global spi


    if (isTracking != 1):
        print('LOST')
        #BuzzerWarning() #Active buzzer
        #GPIO.output(buzzer_pin, GPIO.HIGH)
    else:
        print('TRACKING')
        timerLost = 0
        #GPIO.output(buzzer_pin, GPIO.LOW)        
    SpiToArm(isTracking, ePosition, eDistance)

def timerThreads(buzzer_pin, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer):
    global spi
    global ePosition
    global eDistance
    global isTracking
    global frame
    global sio
    
    try:  
        #SPI config
        spi = spidev.SpiDev() 
        spi.open(0,0)          #Port 0, device 0 (cs0)
        spi.max_speed_hz = 16000000
        
        sio.connect("https://suitcase-server.herokuapp.com")
        
        tmr1 = TimerClass.Interval(1, Task1s)
        tmr1.start()
        tmr100 = TimerClass.Interval(0.1, Task100ms)
        tmr100.start()    
        while True:        
            #Get 4 impor,,tant variable in process
            #isTracking and frame are always available
            isTracking = isTrackingBuffer.get() if (not isTrackingBuffer.empty()) else 0
            frame = frameBufferToServer.get() if (not frameBufferToServer.empty()) else None
            ePosition = ePositionBuffer.get() if (not ePositionBuffer.empty()) else -1
            eDistance = eDistanceBuffer.get() if (not eDistanceBuffer.empty()) else -1
            
            #print("isTracking", isTracking)
            #print("eDistance", eDistance)
            #print("ePosition", ePosition)
            sleep(0.05)    

    except:
        import traceback
        traceback.print_exc()


#----------------------Main Function----------------------------------
def main():
    #Initialization
    # Image Input
    parser = argparse.ArgumentParser()
    parser.add_argument('-vf','--video', dest = 'video_file_path', default = "",help='Path to input video file. (Default="")')
    args = parser.parse_args()
    video_file_path = args.video_file_path

    #Server
    #init()

    global capturedTime
    global startLost
    #global tmr1
    #global tmr100
    global buzzer_pin

    capturedTime = "2000-01-01 00:00:00"    
    #sio.connect("https://suitcase-server.herokuapp.com")

    # timerThreads()
    try:
        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()
        isTrackingBuffer = mp.Queue(1)
        frameBufferToServer = mp.Queue(1)
        ePositionBuffer = mp.Queue(1)
        eDistanceBuffer = mp.Queue(1)
        #timerThreads(buzzer_pin, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer)

        # Process No.1: Start streaming
        p = mp.Process(target = camThread,
                       args = (LABELS, results, frameBuffer, vidfps, video_file_path, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer),
                       daemon = True)
        p.start()
        processes.append(p)

        # Process No.2: Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target = inferencer,
                       args = (results, frameBuffer, number_of_ncs),
                       daemon = True)
        p.start()
        processes.append(p)
    
        # #Process No.3: Transfer to Server
        p = mp.Process(target = timerThreads,
                       args = (buzzer_pin, isTrackingBuffer, frameBufferToServer, ePositionBuffer, eDistanceBuffer,),
                       daemon = True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
        
    finally:
        for p in range(len(processes)):
            processes[p].terminate()
        print("\n\nFinished\n\n")

if __name__ == '__main__':
    main()