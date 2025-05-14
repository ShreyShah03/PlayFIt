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
r_up = cv2.imread('R_up.png')
l_up = cv2.imread('L_up.png')

# Check if images are loaded correctly
if any(image is None for image in [face_image, left_hand_image, right_hand_image, chest_image, leg_image, r_up, l_up]):
    print("Error loading images. Check the paths.")
    exit(1)

# Exercise list
exercise_list = ['Hands Up', 'Hands on Side', 'Palm Rotation']
current_exercise = random.choice(exercise_list)

# Flags and timers
posture_correct = False  # Assume incorrect initially
show_correct_message = False
correct_message_start_time = None
show_incorrect_message = False
incorrect_message_start_time = None
MESSAGE_DISPLAY_DURATION = 2  # "Correct!" message duration
INCORRECT_DISPLAY_DURATION = 1  # "Incorrect!" message duration

# Function to overlay an image on top of the frame
def overlay_image_alpha(frame, image, x, y):
    h, w = image.shape[:2]
    roi = frame[y:y+h, x:x+w]
    frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, image, 0.5, 0)

# Function to rotate an image around its center
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Function to draw the full-body animated creature
def draw_creature(frame, exercise, palm_angle=0):
    head_center = (100, 165)
    chest_pos = (100, 250)
    right_leg_pos = (100, 344)

    overlay_image_alpha(frame, cv2.resize(face_image, (60, 100)), head_center[0], head_center[1])
    overlay_image_alpha(frame, cv2.resize(chest_image, (60, 100)), chest_pos[0], chest_pos[1])
    overlay_image_alpha(frame, cv2.resize(leg_image, (60, 100)), right_leg_pos[0], right_leg_pos[1])

    if exercise == 'Hands Up':
        overlay_image_alpha(frame, cv2.resize(l_up, (30, 80)), 80, 190)
        overlay_image_alpha(frame, cv2.resize(r_up, (30, 80)), 145, 190)

    elif exercise == 'Palm Rotation':
        overlay_image_alpha(frame, rotate_image(cv2.resize(left_hand_image, (20, 60)), palm_angle), 80, 250)
        overlay_image_alpha(frame, rotate_image(cv2.resize(right_hand_image, (20, 60)), palm_angle), 150, 250)

    elif exercise == 'Hands on Side':
        overlay_image_alpha(frame, cv2.resize(left_hand_image, (30, 60)), 75, 250)
        overlay_image_alpha(frame, cv2.resize(right_hand_image, (30, 60)), 155, 250)

# Function to check user posture
def check_user_posture(results, exercise):
    if not results.pose_landmarks:
        return False  # If no pose detected, return False

    landmarks = results.pose_landmarks.landmark

    if exercise == 'Hands Up':
        return (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y and
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

    elif exercise == 'Hands on Side':
        return (abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x) > 0.15 and
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) > 0.15)

    elif exercise == 'Palm Rotation':
        return (abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y) < 0.1 and
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) < 0.1)

    return False

# Main loop
palm_angle = 0
palm_rotation_direction = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    # Animate palm rotation
    if current_exercise == 'Palm Rotation':
        palm_angle += 5 * palm_rotation_direction
        if palm_angle >= 30 or palm_angle <= -30:
            palm_rotation_direction *= -1

    draw_creature(frame, current_exercise, palm_angle)

    # Check user posture
    correct_posture = check_user_posture(result, current_exercise)

    if correct_posture:
        if posture_correct != True:  # First time correct
            posture_correct = True
            show_correct_message = True
            correct_message_start_time = time.time()
            show_incorrect_message = False  # Hide incorrect message
    else:
        if posture_correct != False:  # First time incorrect
            posture_correct = False
            show_incorrect_message = True
            incorrect_message_start_time = time.time()
            show_correct_message = False  # Hide correct message

    # Always display at least one message
    if show_correct_message:
        cv2.putText(frame, "Correct!", (frame.shape[1] // 2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        if time.time() - correct_message_start_time >= MESSAGE_DISPLAY_DURATION:
            show_correct_message = False
            current_exercise = random.choice(exercise_list)  # Move to next exercise

    elif show_incorrect_message:
        cv2.putText(frame, "Incorrect!", (frame.shape[1] // 2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        if time.time() - incorrect_message_start_time >= INCORRECT_DISPLAY_DURATION:
            show_incorrect_message = False

    else:
        # If no messages, force an incorrect message (for debugging)
        cv2.putText(frame, "Keep Trying!", (frame.shape[1] // 2 - 120, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4, cv2.LINE_AA)

    # Display exercise name
    cv2.putText(frame, f"Exercise: {current_exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Exercise Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
