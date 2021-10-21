import cv2
import pyttsx3
print("Loading assets....")
engine = pyttsx3.init()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
classNames= [];
classFile = "./coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = "./ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "./frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    print("Detecting Objects...")
    foundObjects = [];
    success,image = camera.read()
    classIds, confs, bbox = net.detect(image,confThreshold=0.60,nmsThreshold=0.13)
    
    if len(classIds) != 0:
        for classId in classIds.flatten():
            className = classNames[classId - 1]
            foundObjects.append(className)

    for i in foundObjects:
        print("I found a ",i, "!")
        engine.say(i)
        engine.runAndWait()

