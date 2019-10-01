from imageai.Detection import ObjectDetection
import os

detector = ObjectDetection()
cur_path = os.curdir

model_path = os.path.join(cur_path, "models/yolo-tiny.h5")
input_path = os.path.join(cur_path, "input/test.jpg")

output_path = os.path.join(cur_path,"output/output.jpg")

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])

