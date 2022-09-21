import cv2
import mediapipe as mp
camera=cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
mp_join=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
tipIDs=[8,12,16,20]
def drawHandLandmarks(image,hand_landmarks):
    if hand_landmarks:
        for hand in hand_landmarks:
            mp_join.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)

def countfingers(image,hand_landmarks):
    if hand_landmarks:
        landmarks=hand_landmarks[0].landmark
        fingers=[]
        for id in tipIDs:
            fingertipY=landmarks[id].y
            fingerbottomY=landmarks[id-2].y
            if fingertipY<fingerbottomY:
                fingers.append(1)
            else:
                fingers.append(0)
        totalfingers=fingers.count(1)
        text=f'Fingers :{totalfingers}'
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
while True:
    success,image=camera.read()
    image=cv2.flip(image,1)
    results=hands.process(image)
    hand_landmarks=results.multi_hand_landmarks
    drawHandLandmarks(image,hand_landmarks)
    countfingers(image,hand_landmarks)
    cv2.imshow('mediacontroller',image)
    if cv2.waitKey(1)==32:
        break
cv2.destroyAllWindows()