import pandas as pd

from database_operations import Database
from feature_engineering.preprocessor import Preprocessor
from logger import LogMe
from train_test_split import DataSplit
from feature_selection.feature_select import Features
from storing_model import StoreModel
from Models.clustering import Clusters
from Models.classification_models import Model
from Models.model_evaluation import ModelEvaluation
from Models.model_selection import ModelSelection


class Training:

    def __init__(self, df):
        self.log_write = LogMe()
        self.file_object = open("Logs/training_log.txt", "a+")
        self.store = StoreModel()
        self.df = df

    def train_model(self):
        """
         Method name: train_model.
         Description: This method is going to train the model and give the best model in every clusters.
         """
        try:
            preprocess = Preprocessor()
            self.df = preprocess.duplicate(self.df)
            self.df = preprocess.null_values(self.df)

            df1 = self.df.copy()
            # print(df1.columns)
            # print(len(df1.columns))
            if len(df1.columns) > 10:
                drop_col = "phishing"
                train_test = DataSplit(self.df, "phishing", drop_col)
                x_train = train_test.get_x_train()
                y_train = train_test.get_y_train()

                columns = Features()
                x_train = columns.mutual_information(x_train, y_train)
                names = columns.correlation(x_train, 0.86)  # We are using 86% threshold to check multi-colinearity
                x_train = x_train.drop(columns=names)
                column_names = ["directory_length", "qty_slash_url", "qty_dot_directory", "file_length",
                                "qty_hyphen_directory", "qty_percent_file", "qty_hyphen_file",
                                "qty_underline_directory"]
                # print(column_names)
                # print(len(column_names))
                # column_names.insert(0, "Cluster_no")
                # column_names.insert(-1, "phishing")
                # print(column_names)
                # store = StoreModel()
                # store.list_store("important_columns", column_names)
                df2 = df1[column_names].copy()

            else:
                df2 = df1.copy()
            try:
                df2.drop(columns=["phishing"], inplace=True)
            except Exception as e:
                pass

            cluster = Clusters()
            # print(df2.columns)
            df2 = cluster.k_mean_plus(df2)
            df3 = self.df["phishing"].copy()
            # print(df2.columns)
            df4 = pd.concat([df2, df3], join="outer", axis=1)
            drop_col = "phishing"
            target = ["phishing", "Cluster_no"]
            # df4.to_csv("cleaned_data.csv",index=False)
            train_test = DataSplit(df4, target, drop_col)
            x_train = train_test.get_x_train()
            x_test = train_test.get_x_test()
            y_train = train_test.get_y_train()
            y_test = train_test.get_y_test()
            # print(y_test)
            # print(set(df4.Cluster_no))
            for i in range(len(set(df4.Cluster_no))):
                X_train = x_train[x_train["Cluster_no"] == i].copy()
                X_train.drop(columns="Cluster_no", inplace=True)
                Y_train = y_train[y_train["Cluster_no"] == i].copy()
                Y_train.drop(columns="Cluster_no", inplace=True)

                X_test = x_test[x_test["Cluster_no"] == i].copy()
                X_test.drop(columns="Cluster_no", inplace=True)
                Y_test = y_test[y_test["Cluster_no"] == i].copy()
                Y_test.drop(columns="Cluster_no", inplace=True)

                X_train = preprocess.scale_x_train(X_train)
                X_test = preprocess.scale_x_test(X_test)

                model = Model(X_train, Y_train)
                dt = model.decision_tree()
                result_dt = model.score(dt, X_test, Y_test)
                self.log_write.logger(self.file_object, f"dt{i} = {result_dt}")
                self.store.save_model(f"decision_tree_cluster{i}", dt)
                log_reg = model.logistic_regression()
                result_log_reg = model.score(log_reg, X_test, Y_test)
                self.log_write.logger(self.file_object, f"log_reg{i} = {result_log_reg}")
                self.store.save_model(f"logistic_regression_cluster{i}", log_reg)
                svc = model.support_vector()
                result_svc = model.score(svc, X_test, Y_test)
                self.log_write.logger(self.file_object, f"svc{i} = {result_svc}")
                self.store.save_model(f"support_vector_classifier_cluster{i}", svc)
                rf = model.random_forest()
                result_rf = model.score(rf, X_test, Y_test)
                self.log_write.logger(self.file_object, f"rf{i} = {result_rf}")
                self.store.save_model(f"random_forest_cluster{i}", rf)
                ab = model.ada_boosting()
                result_ab = model.score(ab, X_test, Y_test)
                self.log_write.logger(self.file_object, f"ab{i} = {result_ab}")
                self.store.save_model(f"ada_boosting_cluster{i}", ab)
                gb = model.gradient_boosting()
                result_gb = model.score(gb, X_test, Y_test)
                self.log_write.logger(self.file_object, f"gb{i} = {result_gb}")
                self.store.save_model(f"gradient_boosting_cluster{i}", gb)

            self.log_write.logger(self.file_object, f"Model Built succesfully!")

            eval = ModelEvaluation()
            cluster_model = {}
            for i in range(len(set(df4.Cluster_no))):
                dt = self.store.read_model(f"decision_tree_cluster{i}")
                m1 = eval.get_recall(dt, X_test, Y_test)
                log_reg = self.store.read_model(f"logistic_regression_cluster{i}")
                m2 = eval.get_recall(log_reg, X_test, Y_test)
                svc = self.store.read_model(f"support_vector_classifier_cluster{i}")
                m3 = eval.get_recall(svc, X_test, Y_test)
                rf = self.store.read_model(f"random_forest_cluster{i}")
                m4 = eval.get_recall(rf, X_test, Y_test)
                ab = self.store.read_model(f"ada_boosting_cluster{i}")
                m5 = eval.get_recall(ab, X_test, Y_test)
                gb = self.store.read_model(f"gradient_boosting_cluster{i}")
                m6 = eval.get_recall(gb, X_test, Y_test)

                model_dict = {f"decision_tree_cluster{i}": m1, f"logistic_regression_cluster{i}": m2,
                              f"support_vector_classifier_cluster{i}": m3, f"random_forest_cluster{i}": m4,
                              f"ada_boosting_cluster{i}": m5, f"gradient_boosting_cluster{i}": m6}

                best_model = ModelSelection()
                select_model = best_model.best_model_score(model_dict)
                cluster_model["cluster" + f"{i}"] = select_model

            return cluster_model

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building decision_tree {e}")


if __name__ == '__main__':
    try:
        path = r"dataset_full.csv"
        # path = ""
        df = pd.read_csv(path)
    except Exception as e:
        data = Database()
        df = data.fetch_data()

    train_me = Training(df)
    cluster_model = train_me.train_model()
