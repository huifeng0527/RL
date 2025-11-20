import cv2


class CameraStream:
    def __init__(self,  width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)




    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to read from camera")
        return frame
    


    def release(self):
        self.cap.release()