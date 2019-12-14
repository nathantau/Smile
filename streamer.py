import cv2
import numpy as np
from prediction import get_trained_model
from prediction import predict_image
from training import convert_to_3_channels, scale_coordinates
from drawer import zip_to_coordinate_pairs, draw
import matplotlib.pyplot as plt

class Streamer():

    def __init__(self):
        print('Initialized')
        self.model = get_trained_model('modelMKIII.h5')

    def stream(self):

        stream = cv2.VideoCapture(0)

        while True:

            ret, frame = stream.read()

            width = frame.shape[0]
            height = frame.shape[1]

            frame = frame[int((width - 96*5)/2) : int(width - (width - 96*5)/2), int((height - 96*5)/2) : int(height - (height - 96*5)/2), :]

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            print(frame.shape)

            frame_to_predict = cv2.resize(frame, (96, 96))
            frame_to_predict = convert_to_3_channels(frame_to_predict)
            frame_to_predict = frame_to_predict.astype(float)

            print(frame_to_predict.shape)

            output = predict_image(self.model, frame_to_predict)
            output = output[0]

            output = scale_coordinates(output, 96.0)
            output = np.array(output)

            zipped_coordinates = zip_to_coordinate_pairs(output)
            print('zipped', zipped_coordinates)

            # In case I want to use the drawing mechanism

            # output = scale_coordinates(output, 1.0/96.0)
            # draw(frame_to_predict, output)

            frame_to_predict = np.transpose(frame_to_predict, (2, 0, 1))[0]

            for coordinate_pair in zipped_coordinates:
                x = int(coordinate_pair[0])
                y = int(coordinate_pair[1])

                for i in range(5):
                    for j in range(5):
                        frame_to_predict[y + i][x + j] = 0

            frame_to_predict = frame_to_predict / 255
            
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 600,600)
            cv2.imshow('frame', frame_to_predict)

            if cv2.waitKey(33) == ord('q'):
                break

        stream.release()
        cv2.destroyAllWindows() 
                
streamer = Streamer()
streamer.stream()