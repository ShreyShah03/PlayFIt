import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV webcam
cap = cv2.VideoCapture(0)

# Load images
face_image = cv2.imread('face.png')
left_hand_image = cv2.imread('R_hand.png')
right_hand_image = cv2.imread('L_hand.png')
chest_image = cv2.imread('chest.png')
leg_image = cv2.imread('leg.png')

# Check if images are loaded correctly
if any(image is None for image in [face_image, left_hand_image, right_hand_image, chest_image, leg_image]):
    print("Error loading images. Check the paths.")
    exit(1)

# Exercise state
exercise_list = ['Hands Up', 'Hands on Side', 'Palm Rotation']
current_exercise = random.choice(exercise_list)

# Variable to store success message time
show_success_message = False
success_message_start_time = None
SUCCESS_MESSAGE_DISPLAY_DURATION = 1  # Show "Great!" for 1 second
SUCCESS_MESSAGE_INTERVAL = 2  # Interval before the success message can be shown again (in seconds)

# Function to overlay an image on top of the frame
def overlay_image_alpha(frame, image, x, y):
    """Overlays `image` on `frame` at position (x, y) with a simple alpha mask."""
    h, w = image.shape[:2]
    roi = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, image, 0.5, 0)

# Function to draw the full body animated creature in the left corner
def draw_creature(frame, exercise):
    # Set positions
    head_center = (100, 200)
    chest_pos = (100, 250)
    left_leg_pos = (80, 330)
    right_leg_pos = (120, 330)
    
    # Draw face
    face_resized = cv2.resize(face_image, (40, 60))
    overlay_image_alpha(frame, face_resized, head_center[0], head_center[1])
    
    # Draw chest
    chest_resized = cv2.resize(chest_image, (60, 100))
    overlay_image_alpha(frame, chest_resized, chest_pos[0], chest_pos[1])
    
    # Draw legs
    left_leg_resized = cv2.resize(leg_image, (40, 80))
    right_leg_resized = cv2.resize(leg_image, (40, 80))
    overlay_image_alpha(frame, left_leg_resized, left_leg_pos[0], left_leg_pos[1])
    overlay_image_alpha(frame, right_leg_resized, right_leg_pos[0], right_leg_pos[1])

    # Draw hands based on the exercise
    if exercise == 'Hands Up':
        left_hand_pos = (80, 180)
        right_hand_pos = (120, 180)
    elif exercise == 'Hands on Side':
        left_hand_pos = (40, 250)
        right_hand_pos = (160, 250)
    elif exercise == 'Palm Rotation':
        left_hand_pos = (80, 220)
        right_hand_pos = (120, 220)
    
    left_hand_resized = cv2.resize(left_hand_image, (30, 60))
    right_hand_resized = cv2.resize(right_hand_image, (30, 60))
    overlay_image_alpha(frame, left_hand_resized, left_hand_pos[0], left_hand_pos[1])
    overlay_image_alpha(frame, right_hand_resized, right_hand_pos[0], right_hand_pos[1])

# Function to check if the user is performing the current exercise
def check_user_posture(results, exercise):
    landmarks = results.pose_landmarks.landmark

    if exercise == 'Hands Up':
        left_hand_up = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_hand_up = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        return left_hand_up and right_hand_up

    elif exercise == 'Hands on Side':
        left_hand_side = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x) > 0.15
        right_hand_side = abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) > 0.15
        return left_hand_side and right_hand_side

    elif exercise == 'Palm Rotation':
        left_hand_in_position = abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y) < 0.1
        right_hand_in_position = abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) < 0.1
        return left_hand_in_position and right_hand_in_position

    return False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to correct the mirrored image
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # Draw the creature performing the exercise in the bottom-left corner
    draw_creature(frame, current_exercise)

    # Check if user is performing the current exercise
    if result.pose_landmarks:
        if check_user_posture(result, current_exercise):
            if not show_success_message:
                show_success_message = True
                success_message_start_time = time.time()

            current_time = time.time()
            if show_success_message:
                if current_time - success_message_start_time >= SUCCESS_MESSAGE_DISPLAY_DURATION:
                    show_success_message = False
                    while current_time - success_message_start_time < SUCCESS_MESSAGE_INTERVAL:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = pose.process(frame_rgb)
                        draw_creature(frame, current_exercise)
                        cv2.putText(frame, "Great!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, f"Exercise: {current_exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Exercise Tracker', frame)
                        cv2.waitKey(1)
                        current_time = time.time()

                    current_exercise = random.choice(exercise_list)

    if show_success_message:
        cv2.putText(frame, "Great!", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

  
    cv2.putText(frame, f"Exercise: {current_exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Exercise Tracker', frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
