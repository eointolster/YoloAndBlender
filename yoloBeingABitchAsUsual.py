import socket
import json
import cv2
from ultralytics import YOLO
import time

# Initialize the YOLO model
model = YOLO('yolov8n-pose.pt')

# Set up the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 12345))
server_socket.listen()

print("Waiting for connection...")
client_socket, addr = server_socket.accept()
print("Connected to Blender")
client_socket.settimeout(0.1)  # Set a timeout of 0.1 seconds

# Start the video capture
capture = cv2.VideoCapture(0)

# Body part labels
index_to_label = {
    0: 'Nose', 1: 'Eye.L', 2: 'Eye.R', 3: 'Ear.L', 4: 'Ear.R',
    5: 'Shoulder.L', 6: 'Shoulder.R', 7: 'Elbow.L', 8: 'Elbow.R',
    9: 'Wrist.L', 10: 'Wrist.R', 11: 'Hip.L', 12: 'Hip.R',
    13: 'Knee.L', 14: 'Knee.R', 15: 'Ankle.L', 16: 'Ankle.R'
}

try:
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        # Run the model on the frame
        persons = model(frame)
        
        # Process each person's keypoints
        for results in persons:
            for result in results:
                if hasattr(result, 'keypoints'):
                    # Extract keypoints
                    kpts = result.keypoints.xy.cpu().numpy()
                    keypoints_list = kpts.flatten().tolist()

                    # Attach labels to keypoints
                    labels = [index_to_label.get(i, '') for i in range(len(keypoints_list) // 2)]

                    # Draw keypoints and labels
                    for i, (x, y) in enumerate(zip(keypoints_list[::2], keypoints_list[1::2])):
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Draw keypoint
                        label = labels[i]
                        if label:
                            cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Send data to Blender
                    keypoints_data = json.dumps({'keypoints': keypoints_list, 'labels': labels})
                    try:
                        print("Sending data:", keypoints_data)
                        client_socket.sendall(keypoints_data.encode('utf-8'))
                    except socket.error as e:
                        print(f"Send error: {e}")
                        break  # Exit the loop if send fails
                    time.sleep(0.1)  # Delay of 0.1 seconds

                    # Non-blocking receive with timeout
                    try:
                        ack = client_socket.recv(8192).decode('utf-8')
                        if ack == 'q':
                            print("End command received")
                            break
                    except socket.timeout:
                        pass  # No data received, continue with the loop

        cv2.imshow('YOLO Keypoints', frame)
        if cv2.waitKey(1) == ord('q'):  # Quit if 'q' is pressed
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Release resources
    capture.release()
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Connection closed")
