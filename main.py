import cv2
import numpy as np
from scipy.spatial import distance
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.core import Detections
from ultralytics import YOLO


class DetectorTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.class_names_dict = self.model.model.names
        self.box_annotator = BoxAnnotator()
        self.point = ()
        self.tracking = False
        self.track_id = None

    def __clickEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = (x, y)

    def __select_box(self, bboxes):
        if len(bboxes) != 0:
            x, y = self.point
            min_distance = None
            bbox_track = []

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                if (x1 <= x <= x2) and (y1 <= y <= y2):
                    center = ((x2 + x1) / 2, (y2 + y1) / 2)
                    dist = distance.euclidean(center, self.point)
                    if min_distance:
                        if dist < min_distance:
                            min_distance = dist
                            bbox_track = bbox
                    else:
                        min_distance = dist
                        bbox_track = bbox
            return bbox_track
        else:
            return []

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.__clickEvent)
        tracker_id = None
        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            results = self.model.track(frame, tracker="botsort.yaml", persist=True)[0]

            xyxy = results.boxes.xyxy.cpu().numpy()
            confidence = results.boxes.conf.cpu().numpy()
            class_id = results.boxes.cls.cpu().numpy().astype(int)

            if results.boxes.id is not None:
                tracker_id = results.boxes.id.cpu().numpy().astype(int)
            else:
                tracker_id = None

            if self.point != ():
                box_track = self.__select_box(bboxes=detections.xyxy)
                idx = np.where(detections.xyxy == box_track)[0][0]

                self.track_id = tracker_id[idx]
                self.tracking = True
                self.point = ()

            if self.tracking:
                idx = np.where(tracker_id == self.track_id)
                if len(idx[0]) != 0:
                    idx = idx[0][0]
                    xyxy = np.array([xyxy[idx]])
                    confidence = np.array([confidence[idx]])
                    tracker_id = np.array([tracker_id[idx]])
                    class_id = np.array([class_id[idx]])
                else:
                    xyxy = np.empty((0, 4), 'float64')
                    confidence = np.array([], 'float64')
                    tracker_id = np.array([], 'int')
                    class_id = np.array([], 'int')

            detections = Detections(xyxy=xyxy,
                                    confidence=confidence,
                                    class_id=class_id,
                                    tracker_id=tracker_id)

            labels = [
                f"{tracker_id}: {self.class_names_dict[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, tracker_id
                in detections
            ]

            frame = self.box_annotator.annotate(scene=frame,
                                                detections=detections,
                                                labels=labels)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    det_track = DetectorTracker()
    det_track.run()
