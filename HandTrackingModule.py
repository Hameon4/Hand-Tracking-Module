import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=False, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands  # create an object from class hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.track_confidence)

        self.mp_draw = mp.solutions.drawing_utils  # draw the hand schematic on actual hand in webcam

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to rgb cuz object 'hands' uses rgb
        self.results = self.hands.process(img_rgb)  # processing the image and give the result
        # print(results.multi_hand_landmarks)   # used to check if something is detected or not

        if self.results.multi_hand_landmarks:  # check if something is detected or not
            for hand_lm in self.results.multi_hand_landmarks:  # for each hand landmark in the detected hands list
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        land_mark_list = []
        # check if any landmark is detected or not
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, land_mark in enumerate(my_hand.landmark):
                height, width, channel = img.shape
                cx, cy = int(land_mark.x * width), int(land_mark.y * height)
                # print(id, cx, cy)
                land_mark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return land_mark_list


def main():
    cap = cv2.VideoCapture(0)
    # making the frame rate
    previous_time = 0
    current_time = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        land_mark_list = detector.find_position(img)
        if len(land_mark_list) != 0:
            print(land_mark_list[8])
        # calculate the fps
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # insert and display the fps value onto the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
