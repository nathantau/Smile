import pandas as pd

class CsvReader():
    '''
    This is a utility class with the sole purpose of
    making it easier to extract the necessary components
    of a *csv file containing the coordinates of facial
    features.
    '''
    def __init__(self, file_path: str):
        '''
        Uses pandas to read *csv at the specified file path
        '''
        self.df = pd.read_csv(file_path)

    def get_data(self):
        '''
        Returns the whole dataframe passed into the reader on
        construction
        '''
        return self.df

    def get_facial_features(self):
        '''
        Returns a dataframe with the important facial features
        specified.
        '''
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