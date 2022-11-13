from sklearn.cluster import KMeans
from logger import LogMe
from storing_model import StoreModel


class Clusters:
    def __init__(self):
        self.log_write = LogMe()
        self.file_object = open("Logs/Clusters_log.txt", "a+")
        self.store = StoreModel()

    def k_mean_plus(self, df):
        """
         Method name: k_mean_plus.
         Description: This method is going to create a clusters for classifier models.
         df : dataframe
         """
        try:
            k_cluster = KMeans(n_clusters=2, init="k-means++", random_state=100)
            k_cluster.fit(df)
            df["Cluster_no"] = k_cluster.predict(df)
            self.store.save_model("k_means", k_cluster)
            return df
        except Exception as e:
            self.log_write.logger(self.file_object, f"Error in making clusters {e}")
