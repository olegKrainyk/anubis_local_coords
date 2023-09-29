import math
from ultralytics import YOLO
import cv2
import json
model = YOLO("yolov8n.pt")
results = model.predict(source="./distance_test21.mov", show=True, save=False, stream=True)


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

# creating an empty canvas
fig = plt.figure()
ax = plt.axes(projection="3d")


file1 = open(r"./data.txt", "w+")
file1.truncate(0)

xData = []
yData = []
zData = []

for result in results:
        boxes = result.boxes.cpu().numpy()
        # print(result.boxes.xywh)                                 # box with xywh format, (N, 4)


        class object:
            def __init__(self, name, x, y, w, h, X, Y, Z):
                self.name = name
                self.x = x
                self.y = y
                self.w = w
                self.h = h
                self.X = X
                self.Y = Y
                self.Z = Z


        objects = []
        data = []


        def get_local_cords(classification, bearing_center, sensor_width_px, sensor_height_px, target_center_x, target_center_y,
                            traget_width_px, target_height_px, focal_length, sensor_width):

            sensor_height = sensor_width * (sensor_height_px / sensor_width_px)

            #  sensor_height_px height of the video

            # get bearing from two points (for future UI uasge)

            bearing_center_deg = (bearing_center * 180 / math.pi + 360) % 360

            # Get probable height, use 1 meter as default if not in probable_heights
            # probable_height = probable_heights.get(classification, 1.0)
            probable_height = 1.7

            height_on_sensor = (sensor_height * target_height_px) / sensor_height_px

            field_of_view = math.degrees(2 * math.atan((sensor_width / 2) / focal_length))

            distance_to_object = (probable_height * focal_length) / height_on_sensor

            target_bearing_deg = (bearing_center_deg - (field_of_view / 2)) + (
                        (target_center_x) / (sensor_width_px / field_of_view))
            target_bearing = math.radians(target_bearing_deg)

            targetX = distance_to_object * math.cos(target_bearing)
            targetY = distance_to_object * math.sin(target_bearing)


            zconst = ((sensor_height_px / target_height_px) * probable_height)/sensor_height_px

            if((sensor_height_px / 2) < target_center_y):
                targetZ = (sensor_height_px - target_center_y) * zconst
            else:
                targetZ = ((sensor_height_px/2)-target_center_y) * zconst * -1


            print(targetX, targetY, targetZ)

            return ([targetX, targetY, targetZ])


        # img = cv2.VideoCapture('./distance_test21.mov')
        # height_of_sensor_px = img.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # width_of_sensor_px = img.get(cv2.CAP_PROP_FRAME_WIDTH)

        for box in boxes:                                            # iterate boxes
            print("parameters of: ", result.names[int(box.cls[0])])
            # r = box.xywh[0].astype(int)
            xywh = box.xywh[0]

            o1 = object(result.names[int(box.cls[0])], xywh[0], xywh[1], xywh[2], xywh[3], 0, 0, 0)
            print("x-center:", o1.x, " y-center:", o1.y, " width:", o1.w, "px  height:", o1.h)
            # ( height of sensor (mm) * height of object(px) ) / height of sensor (px)
            cords = get_local_cords(o1.name, 90, 4032, 3024, o1.x, o1.y, o1.w, o1.h, 6.86, 10.16)

            o1.X = cords[0]
            o1.Y = cords[1]
            o1.Z = cords[2]

            xData.append(cords[0])
            yData.append(cords[1])
            zData.append(cords[2])

            data.append({"id": f"{len(objects)}", "name": f"{o1.name}", "X": f"{o1.X}", "Y": f"{o1.Y}", "Z": f"{o1.Z}"})

        file1.write("{'objects':")
        file1.write(str(data))
        file1.write('}')
file1.close()

print(xData, yData, zData)

ax.plot3D(xData, yData, zData, 'red')

ax.scatter3D(xData, yData, zData, c=zData, cmap='cividis');

# Showing the above plot
plt.show()






# get_local_cords('car', 90, 4032, 3024, 973.1006, 207.34973, 6.86, 10.16)



