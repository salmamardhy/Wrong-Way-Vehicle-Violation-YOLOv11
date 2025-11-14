# Wrong-Way Vehicle Violation Detection Based on Video Footage Using the You Only Look Once (YOLO) Algorithm

Roads serve as crucial infrastructure for public mobility and goods distribution. 
However, unmanaged and poorly supervised road usage can jeopardize user safety. This study aims to develop an effective wrong way vehicle violation detection system based on video imagery by utilizing object detection and segmentation models from YOLOv11 nano version, CLAHE image enhancement, and the ByteTrack tracking method. CCTV recordings from various locations and public datasets were used for training. 

The YOLOv11n model achieved high performance with a mean Average Precision (mAP) of 97.2% (bounding box) and 95.9% (mask) for road segmentation, and 91.6% for vehicle detection. Violation detection was conducted using cosine similarity and Euclidean distance between the vehicle motion vectors and the dominant traffic direction vector, with thresholds of 0.7 for cosine similarity and a minimum Euclidean distance of 5 pixels. 

Experimental results show that the system accurately detects violations, particularly during daytime, achieving 17.40 FPS, 99.70% accuracy, 99% recall, and 0.49% FPR. At night, accuracy decreased to 98.93% and recall 92,24% with an FPR of 2.5% due to poor lighting conditions. These results indicate that the developed system can reliably detect wrong-way driving violations with low error rates and high performance, making it suitable to support automated traffic surveillance.

=================================== SETUP ===================================

📌 Features
1) Vehicle detection using YOLOv11
2) Road area segmentation
3) Wrong-way vehicle direction classification
4) Multi-object tracking with ByteTrack
5) CLAHE preprocessing + BRISQUE image quality scoring
6) FastAPI backend (video upload, streaming, detection)
7) React + Vite + Chakra UI frontend
8) Live video streaming and violation sidebar with auto-refresh

🔧 Backend Setup (FastAPI)
1. Navigate to the backend folder using 
<pre>cd backend </pre>
2. Create and activate a virtual environment
<pre>python -m venv venv
venv\Scripts\activate </pre>
3. Install dependencies
<pre>pip install -r requirements.txt</pre>
4. Start the backend server
<pre>uvicorn main:app --reload --host 0.0.0.0 --port 8000</pre>

🔧 Frontend Setup (React + Vite + Chakra UI)
1. Navigate to the frontend folder using 
<pre>cd frontend</pre>
2. Instal dependencies
<pre>npm install</pre>
3. Start the development server
<pre>npm run dev</pre>

📤 How to Use the System
1. Start both backend and frontend
2. Open the frontend UI in your browser
3. Upload a video (MP4) or enter a stream URL (RTSP/HTTP)
4. Click Start Stream
5. Watch vehicle detections in real time
6. Violation sidebar will update automatically, showing:
- Violation timestamp
- Vehicle ID
- Cropped violation image
- Wrong-way / correct-way status
