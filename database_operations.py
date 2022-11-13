import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class Database:

    def __init__(self):
        cloud_config = {
            'secure_connect_bundle': r'Creadintials\secure-connect-dataset-for-project.zip'
        }
        auth_provider = PlainTextAuthProvider('OIZlpBpNSOKMltxbyqZsYblP',
                                              'j2kQD8r4Z,8qjWYWeQjhhKoDGwd0zb5F9ZwYzN6onS+ZlD'
                                              '-oAZRIsp_vs_HhElfslbDAB6s-_Ko884DE+662Ks4tvKuegZ8zyK2EniSZdh-,,'
                                              'q34mC.C7nd28tRiy5t8')
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = cluster.connect()

    def fetch_data(self):
        """
         Method name: fetch_data.
         Description: This method is going to fetch the data from the database.
         """
        try:
            q = f"select * from dataset.phishing_dataset_cleaned"
            data = self.session.execute(q)
            lst = []
            for i in data.all():
                lst.append(i)
            df = pd.DataFrame(lst)
            df.drop(columns="serial_no", inplace=True)
            return df

        except Exception as e:
            pass
