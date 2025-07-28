import streamlit as st
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")  # thay bằng đường dẫn đến file .pt của bạn

st.title("🟢 Realtime Object Detection with YOLOv11")

# Button để bật/tắt camera
start_camera = st.button("📷 Start Camera")

# Vùng hiển thị video
frame_placeholder = st.empty()

# Nếu nhấn nút
if start_camera:
    cap = cv2.VideoCapture(0)  # dùng camera mặc định

    if not cap.isOpened():
        st.error("Không mở được camera!")
    else:
        st.info("Nhấn Stop (trên thanh điều khiển) để dừng realtime")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Không thể lấy hình ảnh từ camera.")
            break

        # YOLO predict
        results = model.predict(source=frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        # Hiển thị ảnh trên Streamlit
        frame_placeholder.image(annotated_frame, channels="BGR")

    cap.release()