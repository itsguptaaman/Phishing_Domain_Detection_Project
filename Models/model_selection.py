from logger import LogMe


class ModelSelection:
    def __init__(self):
        self.log_write = LogMe()
        self.file_object = open("Logs/Model_selection_log.txt", "a+")

    def best_model_score(self,model_dt):
        """
        Method name: best_model_score.

        Description: This method is going to give me the model with the highest accuracy.

        name: The name of the model you are going to use while saving it.

        model: The object of the model you are going to store..
        """
        try:
            find_max = max(model_dt, key=model_dt.get)
            return find_max
        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while giving output Models {e}")
