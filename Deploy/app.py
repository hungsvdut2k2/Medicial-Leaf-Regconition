import cv2
import torch
from PIL import Image

model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="Deploy/weights/best.pt"
)
video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    image = Image.fromarray(frame)
    results = model([image], 640)
    for i, row in results.pandas().xyxy[0].iterrows():
        xmin, ymin, xmax, ymax = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )
        confidence = float(row["confidence"])
        leaves_type = str(row["name"])
        if confidence >= 0.6:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(
                frame,
                leaves_type + " " + str(round(confidence, 2)),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
