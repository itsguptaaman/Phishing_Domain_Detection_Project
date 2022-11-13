

from sklearn.model_selection import train_test_split
from logger import LogMe


class DataSplit:

    def __init__(self,df, target, drop_col):
        self.log_write = LogMe()
        self.file_object = open("Logs/train_test_split_log.txt", "a+")
        self.df=df
        self.target=target
        self.drop_col = drop_col

    def split(self):
        """
         Method name: split.
         Description: This method is going to split our dataframe into training and testing data
         """
        try:
            dataset = self.df.copy()
            x = dataset.drop(columns=self.drop_col)
            y = dataset[self.target]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
            self.log_write.logger(self.file_object,f"Splitting done succesfully")
            return [x_train, x_test, y_train, y_test]

        except Exception as e:
            self.log_write.logger(self.file_object,f"Error while splitting the data {e}")

    def get_x_train(self):
        return self.split()[0]

    def get_x_test(self):
        return self.split()[1]

    def get_y_train(self):
        return self.split()[2]

    def get_y_test(self):
        return self.split()[3]

