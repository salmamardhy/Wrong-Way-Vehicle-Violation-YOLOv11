# Wrong-Way Vehicle Violation Detection Based on Video Footage Using the You Only Look Once (YOLO) Algorithm

Roads serve as crucial infrastructure for public mobility and goods distribution. 
However, unmanaged and poorly supervised road usage can jeopardize user safety. This study aims to develop an effective wrong way vehicle violation detection system based on video imagery by utilizing object detection and segmentation models from YOLOv11 nano version, CLAHE image enhancement, and the ByteTrack tracking method. CCTV recordings from various locations and public datasets were used for training. 

The YOLOv11n model achieved high performance with a mean Average Precision (mAP) of 97.2% (bounding box) and 95.9% (mask) for road segmentation, and 91.6% for vehicle detection. Violation detection was conducted using cosine similarity and Euclidean distance between the vehicle motion vectors and the dominant traffic direction vector, with thresholds of 0.7 for cosine similarity and a minimum Euclidean distance of 5 pixels. 

Experimental results show that the system accurately detects violations, particularly during daytime, achieving 17.40 FPS, 99.70% accuracy, 99% recall, and 0.49% FPR. At night, accuracy decreased to 98.93% and recall 92,24% with an FPR of 2.5% due to poor lighting conditions. These results indicate that the developed system can reliably detect wrong-way driving violations with low error rates and high performance, making it suitable to support automated traffic surveillance.
