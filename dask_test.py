from dask.distributed import Client, LocalCluster

if __name__ == "__main__":

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    print(client.dashboard_link)
    
    client.close()
    cluster.close()