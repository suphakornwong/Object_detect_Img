import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict

def main():
    st.title("Project Snail Detection")
    st.sidebar.write('<p style="font-size: 18px; font-weight: bold;">งานวิจัยของศุภกร วงษ์เรืองพิบูล</p>', unsafe_allow_html=True)

    model_path = "BestObjectDetect".pt"
    st.sidebar.success("Model loaded successfully...")
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())

        model = YOLO(model_path)
        img = cv2.imread(tfile.name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.sidebar.info("Object Detecting ...")
        results = model(img)

        label_count = defaultdict(int)

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                label = f"{model.names[int(cls)]}"
                label_count[label] += 1

                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_rgb, f"{label} ({conf:.2f})", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(img_rgb, caption="ผลการทำ Object Detection", width="stretch")

        st.sidebar.subheader("สรุปผลการ Detect")
        for label, count in label_count.items():
            st.sidebar.write(f"- **{label}**: {count}")

        st.success("Object Detection Completed")

        try:
            tfile.close()
            os.unlink(tfile.name)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการลบไฟล์ชั่วคราว: {e}")

main()
