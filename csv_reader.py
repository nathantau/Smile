import pandas as pd

class CsvReader():

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def get_data(self):
        return self.data