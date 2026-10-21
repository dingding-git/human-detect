import cv2
from yolov5 import YOLOv5
from deep_sort import DeepSort


class PedestrianDetection:
    def __init__(self, yolo_weights, deepsort_weights):
        self.yolo = YOLOv5(yolo_weights)
        self.deep_sort = DeepSort(deepsort_weights)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_people = 0
        gender_count = {'male': 0, 'female': 0}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.yolo.detect(frame)
            tracks = self.deep_sort.update(detections)

            for track in tracks:
                bbox = track.to_tlbr()
                attributes = self.predict_attributes(frame[bbox])  # 自定义属性识别方法
                total_people += 1
                if attributes['gender'] == 'male':
                    gender_count['male'] += 1
                else:
                    gender_count['female'] += 1

                # 绘制边界框和属性
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{attributes['gender']}",
                            (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"Total people: {total_people}, Male: {gender_count['male']}, Female: {gender_count['female']}")

    def predict_attributes(self, image):
        # 这里调用训练好的属性识别模型进行预测
        # 具体实现取决于你的模型结构和输入要求
        # 返回一个字典，例如：{'gender': 'male'}
        pass
