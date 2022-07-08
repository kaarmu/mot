import cv2
import numpy as np

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

from sort.sort import Sort

LABELS_PATH = 'data/coco.names'
CFG_PATH = 'data/yolov4-tiny.cfg'
WEIGHTS_PATH = 'data/yolov4-tiny.weights'

USE_CUDA = True

MAX_AGE = 1
MIN_HITS = 3
IOU_THRESHOLD = 0.3

def plot_label(image, box, labels, append=''):
    image = np.copy(image)

    height, width, _ = image.shape

    x1, y1, x2, y2, _, cls_conf, cls_id, *_ = box

    x1 = int(x1 * width)
    y1 = int(y1 * height)
    x2 = int(x2 * width)
    y2 = int(y2 * height)

    label = labels[cls_id] if cls_id < len(labels) else '?'

    bbox_thick = int(0.6 * (height + width) / 600)
    color = (255, 0, 0) # tuple(map(int, np.random.rand(3) * 255))
    msg = f'{label} ({cls_conf:.0%}) {append}'
    msg_w, msg_h = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]

    cv2.rectangle(image, (x1, y1), (x1 + msg_w, y1 - msg_h - 3), color, -1)

    image = cv2.putText(image, msg, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, bbox_thick)

    return image

## Set up

labels = load_class_names(LABELS_PATH)

model = Darknet(CFG_PATH)
model.load_weights(WEIGHTS_PATH)

if USE_CUDA:
    model.cuda()

mot_tracker = Sort(max_age=MAX_AGE,
                   min_hits=MIN_HITS,
                   iou_threshold=IOU_THRESHOLD)
trackers = []

# Main

vid_cap = cv2.VideoCapture('/dev/video0')

width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
fourcc = int(vid_cap.get(cv2.CAP_PROP_FOURCC))

while vid_cap.grab():

    _, frame = vid_cap.retrieve()

    detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_frame = cv2.resize(detection_frame, (model.width, model.height))

    boxes = do_detect(model, detection_frame, 0.4, 0.6, USE_CUDA, show_stats=False)[0]

    if boxes:
        dets = np.array(boxes)[:, :4]
        trackers = mot_tracker.update(dets)

    for box in boxes:

        if not box:
            continue

        ## Get box

        # box have normalized u, v
        u1, v1, u2, v2, _, conf, label_id = box

        # get real pixel coords
        u1, u2 = [min(width, max(0, round(u*width))) for u in (u1, u2)]
        v1, v2 = [min(height, max(0, round(v*height))) for v in (v1, v2)]

        if len(trackers):
            vec = np.array([u1, v1, u2, v2, 0])
            track_id = min(trackers, key=lambda t: np.linalg.norm(vec-t))[-1]
        else:
            track_id = -1

        frame = plot_label(frame, box, labels, append=f'id={int(track_id)}')

    print('Following targets', {len(trackers)}, *(t[-1] for t in trackers))
    cv2.imshow('MOT test', frame)

    if cv2.waitKey(10) == ord('q'):
        break


vid_cap.release()
