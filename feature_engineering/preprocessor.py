"""
This is the Entrance of the preprocessor module
"""
from logger import LogMe
from storing_model import StoreModel
from sklearn.preprocessing import StandardScaler
import pandas as pd


class Preprocessor:

    def __init__(self):
        self.log_write = LogMe()
        self.file_object = open("Logs/preprocessing_log.txt", "a+")
        self.store = StoreModel()

    def duplicate(self, df):
        """
        Method name: duplicate.
        Description: This method is going to drop all the duplicate entry in the dataset.
        df :- DataFrame
        """
        try:
            df = df.drop_duplicates()
            return df

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while duplicate values in Dataset {e}")

    def null_values(self, df):
        """
        Method name: null_values.
        Description: This method is going to drop all the nan values  in the dataset.
        df :- DataFrame
        """
        try:
            df = df.dropna()
            return df
        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while handling Null values in Dataset {e}")

    def scale_x_train(self, x_train):
        """
        Method name: scale_x_train.
        Description: This method is going to scale the x_train.
        x_train :- DataFrame or Array.
        """
        try:
            scale = StandardScaler()
            scale.fit(x_train)
            x_train = pd.DataFrame(scale.transform(x_train), columns=x_train.columns)
            self.store.save_model("Scale", scale)
            return x_train

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while performing Standardization {e}")

    def scale_x_test(self, x_test):
        """
        Method name: scale_x_test.
        Description: This method is going to scale the x_test.
        x_test :- DataFrame or Array.
        """
        try:
            scale = self.store.read_model("Scale")
            x_test = pd.DataFrame(scale.transform(x_test), columns=x_test.columns)
            return x_test

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while performing Standardization {e}")
