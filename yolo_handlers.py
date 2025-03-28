import os
import torch
import cv2
from flask import Flask, request, jsonify
from models import db, YoloResult

# YOLO 핸들러 클래스
class YOLOApp:
    def __init__(self):
        self.custom_weights = './pt/yolo.pt'  # 로컬 YOLOv9 가중치 경로

    def detect_video(self, video_path, output_path, stride=10, img_size=640, conf=0.5):
        # 비디오 파일 처리
        try:
            os.system(f"python3 ./yolov9/detect.py --weights {self.custom_weights} --vid-stride {stride} \
                        --img {img_size} --conf {conf} --exist-ok --source {video_path} --save-crop --project {output_path}")
            print(f"비디오 파일 {video_path} 처리가 완료되었습니다.")
        except Exception as e:
            print(f"비디오 처리 중 오류 발생: {e}")

# YOLOAPP 인스턴스 생성
yolo_app = YOLOApp()

def handle_yolo_predict(video_id):
    torch.cuda.empty_cache() 

    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected for uploading"}), 400

    # 파일 저장 경로 설정
    file_path = os.path.join("./uploaded_videos", file.filename)

    # 파일 처리
    if file.filename.endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv')):
        try:
            # YOLOv9 모델을 사용하여 이미지 처리
            out_put_path = "./mp4_to_img"
            yolo_app.detect_video(file_path, out_put_path)

            # 처리된 이미지 저장 경로
            result_image_path = os.path.join('./mp4_to_img', "exp", "crops", "glasses")

            padded_image_path = os.path.join(result_image_path, f"padded")
            os.makedirs(padded_image_path, exist_ok=True)  # exist_ok=True는 이미 폴더가 있으면 에러를 방지합니다.

            for filename in os.listdir(result_image_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
            
                    # 특정 이미지에 대해 패딩 추가
                    image_path = os.path.join(result_image_path, filename)  # Example image file name
                    image = cv2.imread(image_path)
                    if image is None:
                        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

                    # RGB 변환
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 패딩 추가
                    padded_image = cv2.copyMakeBorder(
                        image, 160, 160, 380, 380, cv2.BORDER_CONSTANT, value=[255, 255, 255]
                    )

                    # 패딩된 이미지 저장
                    padded_image_path = os.path.join(result_image_path, f"padded", filename)
                    cv2.imwrite(padded_image_path, cv2.cvtColor(padded_image, cv2.COLOR_RGB2BGR))
                    print(f"패딩된 이미지 저장 완료: {padded_image_path}")

                    # 데이터베이스에 결과 저장
                padded_image_path = os.path.join(result_image_path, f"padded")
                detection_result = YoloResult(
                    video_code=video_id,
                    yolo_result_path=padded_image_path
                )
                db.session.add(detection_result)
                db.session.commit()

            return jsonify({
                "message": "Image processed successfully",
                "yolo_result_code": detection_result.yolo_result_code,
                "output_image": padded_image_path
            }), 200
        except Exception as e:
            return jsonify({"message": f"Error during processing: {str(e)}"}), 500
    else:
        return jsonify({"message": "Unsupported file format. Only MP4, AVI, MKV, MOV, WMV are supported."}), 400
