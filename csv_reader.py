import pandas as pd

class CsvReader():

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def get_data(self):
        return self.df

    def get_relevant_data(self):
        return self.df[[
            'left_eye_center_x',
            'left_eye_center_y',
            'right_eye_center_x',
            'right_eye_center_y',
            'nose_tip_x',
            'nose_tip_y',
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y'
        ]]