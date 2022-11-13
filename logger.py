from datetime import datetime
import os


class LogMe:

    def __init__(self):
        try:
            os.mkdir("Logs")
        except Exception as e:
            pass

    def logger(self, file_object, log_message):
        """
        Method name: Logger.

        Description: This method is going to print all the log messages in
        the file which will be passed by the user.

        file_object: File object is the name of the file.

        log_message: The message which you want to print.

        examples:

        >>> file_object=open("Loggingtest.txt","a+")
        >>> log_write=LogMe()
        >>> log_write.logger(file_object,"This is log example")
        """
        self.now = datetime.now()  # datetime.now will give current date and time
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")
