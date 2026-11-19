from mosaicolabs import MosaicoClient, IMU, QueryOntologyCatalog, Query, QuerySequence
import client_df_parsing

with MosaicoClient.connect("localhost", 6726) as client:
    std = 0.03 # Standard deviation from imu vector3 covariance (variance 0.0009 on the diagonal)
    thr = 5*std # Account for sensor noise
    clustering_dt_ns = 10_000_000 # 10 ms = 2.5 IMU samples (250 Hz): bridges single noisy samples

    # Define query object with matching name and filters
    query = Query(
        QuerySequence().with_name_match("spot_yb42"),
        QueryOntologyCatalog()
            .with_expression(IMU.Q.acceleration.x.between(-thr,thr))
            .with_expression(IMU.Q.acceleration.y.between(-thr,thr))
            .with_expression(IMU.Q.acceleration.z.between(-thr,thr))
    )

    # Server side query
    qresponse = client.query(query=query)

    timestamps = []
    # Inspect the response
    if qresponse is not None:
    # Results are automatically grouped by Sequence for easier data management
        for item in qresponse:
            print(f"Sequence: {item.sequence.name}")
            print(f"Topics: {[topic.name for topic in item.topics]}")
            seq_t0 = client.sequence_handler(item.sequence.name).timestamp_ns_min

            # Clusterize all topics within the sequence to extract the time intervals
            clusters_dict = item.clusterize_all(override_clustering_dt_ns=clustering_dt_ns)

            for t_name, clusters in clusters_dict.items():
                print(f"{t_name}: {len(clusters)} cluster(s)")
                for cluster in clusters:
                    start, end = cluster.timerange.start, cluster.timerange.end
                    delta_t_s = (end - start) * 1e-9
                    print(f"  {cluster}")
                    print(f"  start      = {start} ns")
                    print(f"  start UTC  = {client_df_parsing.ns_to_utc(start).isoformat()}")
                    print(f"  end        = {end} ns ({client_df_parsing.ns_to_utc(end).isoformat()})")
                    print(f"  since seq start = {(start - seq_t0) / 1e9:.3f} s")
                    print(f"  delta_t = {delta_t_s:.3f} s  -> h = g*t^2/2 = {delta_t_s**2 * 9.81 / 2:.3f} m")
                    if not timestamps:  # use the first cluster for the bonus analysis
                        timestamps = [start, end]

if timestamps:
    client_df_parsing.get_df_imu(timestamps)
    client_df_parsing.get_df_loadcells(timestamps)
    client_df_parsing.get_status_log(timestamps)
    client_df_parsing.get_odom(timestamps)
    client_df_parsing.get_tf(timestamps)
    client_df_parsing.get_camera(timestamps)
