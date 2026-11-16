import cv2
from ultralytics import YOLO

# Load your trained YOLOv11 model
# Make sure 'best.pt' is in the same directory
model = YOLO('best.pt')

# Open a connection to the webcam (0 is usually the default cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through webcam frames
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLOv11 inference on the frame
        results = model.predict(source='0', conf=0.75, device='cpu', show=True)

        # Process results
        for r in results:
            # Plot results on the frame
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv11 Face Anti-Spoofing", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the stream ends
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
