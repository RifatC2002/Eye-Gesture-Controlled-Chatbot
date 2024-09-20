
from django.shortcuts import render, reverse
from django.contrib.auth.decorators import login_required
from .models import ChatBot
from django.http import HttpResponseRedirect, JsonResponse
import google.generativeai as genai
import cv2
import mediapipe as mp
import pyautogui

# Create your views here.
# Face Mesh model from mediapipe
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

# Function to capture webcam feed and detect eye gestures
def capture_and_detect_eye_gestures():

    # video input from webcam
    cam = cv2.VideoCapture(0)


    # Capturing frames
    while True:
        _, frame = cam.read()

        # Flipping the frames since camera flips them auto
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecting and extracting landmark points

        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        frame_height, frame_width, _ = frame.shape

        if landmark_points:

            landmarks = landmark_points[0].landmark

            # Right eye
            for id, landmark in enumerate(landmarks[474:478]):

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                cv2.circle(frame, (x, y), 3, (0, 255, 255))

                # Moving cursor
                if id == 1:
                    screen_x = (screen_width * 1.35) * landmark.x
                    screen_y = (screen_height * 1.35) * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            # Left eye
            left_eye_landmarks = [landmarks[145], landmarks[159]]

            for landmark in left_eye_landmarks:

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if (left_eye_landmarks[0].y - left_eye_landmarks[1].y) < 0.004:
                # Mouse click
                pyautogui.click()
                pyautogui.sleep(1)

        cv2.imshow('Eye Gesture Mouse', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# add here to your generated API key
genai.configure(api_key="AIzaSyBFdU58tS3wfisO0uc-cBrVd7M5sDbVQkQ")

@login_required
def ask_question(request):
    if request.method == "POST":
        text = request.POST.get("text")
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat()
        response = chat.send_message(text)
        user = request.user
        ChatBot.objects.create(text_input=text, gemini_output=response.text, user=user)
        # Extract necessary data from response
        response_data = {
            "text": response.text,  # Assuming response.text contains the relevant response data
            # Add other relevant data from response if needed
        }
        return JsonResponse({"data": response_data})
    else:
        return HttpResponseRedirect(
            reverse("chat")
        )  # Redirect to chat page for GET requests


@login_required
def chat(request):
    user = request.user
    chats = ChatBot.objects.filter(user=user)
    import threading
    t = threading.Thread(target=capture_and_detect_eye_gestures)
    t.start()
    return render(request, "chat_bot.html", {"chats": chats})
