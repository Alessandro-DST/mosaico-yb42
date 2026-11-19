import os

from mosaicolabs import MosaicoClient, CompressedImage
from mosaicolabs.ml import DataFrameExtractor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_df(plot_keys, df, fname, ts_vline=[0]):
    """
    Plot subplots for numeric values from a DataFrame

    plot_keys: df keys to plot\n
    df: Pandas DataFrame\n
    fname: output plot file\n
    ts_vline: timestamp vertical lines for plot, by default [0]
    """

    fig, ax = plt.subplots(len(plot_keys)-1, 1)
    for key in plot_keys:
        if key != plot_keys[0]:
            i = plot_keys.index(key)-1
            ax[i].plot(plot_keys[0], key, data=df)
            ax[i].set_title(key)
            ax[i].autoscale(enable=True, axis='both', tight=True)
            ax[i].vlines(ts_vline, df[key].min(), df[key].max(), color='r')
            ax[i].grid()
    fig.set_figwidth(25)
    fig.set_figheight(15)
    fig.savefig(fname)

def get_df_imu(timestamps):
    with MosaicoClient.connect("localhost", 6726) as client:
        sequence_handler = client.sequence_handler("spot_yb42")
        topic_list = ["/imu/body"]
        if sequence_handler:
            frames = []
            for df in DataFrameExtractor(sequence_handler).to_pandas_chunks(
                topics = topic_list
            ):
                df = df.dropna(axis="columns", how="all") # Drop all null columns
                frames.append(df)

            df_concat = pd.concat(frames, ignore_index=True) # Avoid duplicated indexes
            df_concat.info()

            # Select only linear acc data for now
            plot_keys = [
                "timestamp_ns",
                "/imu/body.IMU.acceleration.x",
                "/imu/body.IMU.acceleration.y",
                "/imu/body.IMU.acceleration.z",
                ]

            # Check difference between subsequent row values in DataFrame to identify sampling rate
            df_diff = df_concat[plot_keys].diff()
            print(df_diff.describe())

            plot_df(plot_keys, df_concat, "./plots/lin_acc", timestamps)

            # Select time interval of detected free fall
            df_fall = df_concat[
                (df_concat[plot_keys[0]] >= timestamps[0]) 
                &
                (df_concat[plot_keys[0]] <= timestamps[1])
                ]
            df_fall.info()

            plot_df(plot_keys, df_fall, "./plots/lin_acc_fall", timestamps)

def get_df_loadcells(timestamps):
    with MosaicoClient.connect("localhost", 6726) as client:
        sequence_handler = client.sequence_handler("spot_yb42")
        topic_list = [
            '/contact/FR_foot',
            '/contact/HL_foot',
            '/contact/HR_foot',
            '/contact/FL_foot',
        ]
        if sequence_handler:
            frames = []
            for df in DataFrameExtractor(sequence_handler).to_pandas_chunks(
                topics=topic_list
            ):
                df = df.dropna(axis="columns", how="all") # Drop all null columns
                frames.append(df)

            df_concat = pd.concat(frames, ignore_index=True) # Avoid duplicated indexes
            df_concat.info()

            # Select only linear acc data for now
            plot_keys = ["timestamp_ns"]
            forces = ['.ForceTorque.force.x', '.ForceTorque.force.y', '.ForceTorque.force.z']
            
            for topic in topic_list:
                for force in forces:
                    plot_keys.append("".join([topic, force]))


            # Check difference between subsequent row values in DataFrame to identify sampling rate
           # df_diff = df_concat[plot_keys].diff()
           # print(df_diff.describe())

            plot_df(plot_keys, df_concat, "./plots/ld_forces", timestamps)

            # Select time interval of detected free fall
            df_fall = df_concat[
                (df_concat[plot_keys[0]] >= timestamps[0]) 
                &
                (df_concat[plot_keys[0]] <= timestamps[1])
                ]
            df_fall.info()

            plot_df(plot_keys, df_fall, "./plots/ld_fall", timestamps)

SEQUENCE = "spot_yb42"


def ns_to_utc(ts_ns):
    """
    Convert an integer epoch timestamp in ns to a tz-aware UTC pandas Timestamp.
    pd.Timestamp keeps full nanosecond precision (no float conversion).
    """
    return pd.Timestamp(int(ts_ns), unit="ns", tz="UTC")


def get_topic_df(seq_handler, topic):
    """
    Download a whole topic as a single flattened DataFrame (sequence is small,
    so one big window is fine) and drop the all-empty columns.
    """
    frames = list(
        DataFrameExtractor(seq_handler).to_pandas_chunks(topics=[topic], window_sec=1000)
    )
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(axis="columns", how="all")


def nearest_row(df, ts_ns):
    """Return the row of df whose timestamp_ns is closest to ts_ns."""
    return df.loc[(df["timestamp_ns"] - ts_ns).abs().idxmin()]


def quat_to_yaw_deg(x, y, z, w):
    """Heading (rotation about world z) from a quaternion, in degrees."""
    return np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def get_camera(timestamps, margin_ns=500_000_000, out_dir="./camera_frames"):
    """
    Save the front-camera frames from (fall start - margin) to (fall end + margin)
    as PNG files. The fall window alone (68 ms) is shorter than the camera frame
    interval, so a margin is needed to get frames before and after.

    File names: <index>_<offset from fall start in ms>.png so they sort in time order.
    """
    start_ns, end_ns = timestamps[0], timestamps[1]
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    with MosaicoClient.connect("localhost", 6726) as client:
        seq_handler = client.sequence_handler(SEQUENCE)
        if not seq_handler:
            return saved

        streamer = seq_handler.get_data_streamer(
            ["/camera/front/image"],
            start_timestamp_ns=start_ns - margin_ns,
            end_timestamp_ns=end_ns + margin_ns + 1,  # end is exclusive
        )
        try:
            for i, (topic, msg) in enumerate(streamer):
                img = msg.get_data(CompressedImage)
                if img is None:
                    print(f"[camera] skipped message at {msg.timestamp_ns}: not a CompressedImage")
                    continue

                pil_img = img.to_image()
                if pil_img is None:
                    print(f"[camera] could not decode frame at {msg.timestamp_ns} (format {img.format})")
                    continue

                offset_ms = (msg.timestamp_ns - start_ns) / 1e6
                fname = os.path.join(out_dir, f"{i:03d}_{offset_ms:+.0f}ms.png")
                pil_img.save(fname)
                saved.append((msg.timestamp_ns, offset_ms, fname))
                print(f"[camera] {fname}  t={msg.timestamp_ns}  ({ns_to_utc(msg.timestamp_ns)})")
        finally:
            streamer.close()

    # Contact sheet: all frames side by side, titled by offset from fall start
    if saved:
        fig, axes = plt.subplots(1, len(saved), figsize=(4 * len(saved), 4))
        axes = np.atleast_1d(axes)
        for ax, (_, offset_ms, fname) in zip(axes, saved):
            ax.imshow(plt.imread(fname))
            ax.set_title(f"{offset_ms:+.0f} ms")
            ax.axis("off")
        fig.suptitle("Front camera around the free-fall start (0 ms)")
        fig.tight_layout()
        fig.savefig("./plots/camera_contact_sheet.png")
        plt.close(fig)

    return saved


def get_odom(timestamps):
    """
    Odometry (/robot/odom, MotionState, frame 'world' -> 'body').
    - prints position/heading at the fall start (nearest sample) and right after the fall end
    - plots the top-down trajectory with the fall location marked
    - plots body height z over time with the fall window marked
    """
    start_ns, end_ns = timestamps[0], timestamps[1]
    p = "/robot/odom.MotionState."

    with MosaicoClient.connect("localhost", 6726) as client:
        seq_handler = client.sequence_handler(SEQUENCE)
        if not seq_handler:
            return None
        df = get_topic_df(seq_handler, "/robot/odom")
        t0 = seq_handler.timestamp_ns_min

    df["t_s"] = (df["timestamp_ns"] - t0) / 1e9  # integer subtraction first, then float
    df["yaw_deg"] = quat_to_yaw_deg(
        df[p + "pose.orientation.x"], df[p + "pose.orientation.y"],
        df[p + "pose.orientation.z"], df[p + "pose.orientation.w"],
    )

    print(f"[odom] frame: {df[p + 'header.frame_id'].iloc[0]} -> {df[p + 'target_frame_id'].iloc[0]}, "
          f"{len(df)} samples, mean dt {df['timestamp_ns'].diff().mean() / 1e6:.1f} ms")

    rows = {"fall start": nearest_row(df, start_ns), "fall end": nearest_row(df, end_ns)}
    for label, r in rows.items():
        print(f"[odom] {label}: t={int(r['timestamp_ns'])} ({r['t_s']:.3f} s)  "
              f"x={r[p + 'pose.position.x']:.3f} m  y={r[p + 'pose.position.y']:.3f} m  "
              f"z={r[p + 'pose.position.z']:.3f} m  yaw={r['yaw_deg']:.1f} deg  "
              f"vx={r[p + 'velocity.linear.x']:.2f} vy={r[p + 'velocity.linear.y']:.2f} "
              f"vz={r[p + 'velocity.linear.z']:.2f} m/s")

    # Height around the fall: min/max z in +-1 s
    around = df[(df["timestamp_ns"] >= start_ns - 1_000_000_000) & (df["timestamp_ns"] <= end_ns + 1_000_000_000)]
    print(f"[odom] z in +-1 s around fall: min {around[p + 'pose.position.z'].min():.3f} m, "
          f"max {around[p + 'pose.position.z'].max():.3f} m")
    print(f"[odom] total path length: "
          f"{np.hypot(df[p + 'pose.position.x'].diff(), df[p + 'pose.position.y'].diff()).sum():.1f} m")

    # Top-down trajectory
    fall = rows["fall start"]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(p + "pose.position.x", p + "pose.position.y", data=df, lw=1, label="trajectory")
    ax.scatter(df[p + "pose.position.x"].iloc[0], df[p + "pose.position.y"].iloc[0],
               c="g", s=80, zorder=3, label="start")
    ax.scatter(df[p + "pose.position.x"].iloc[-1], df[p + "pose.position.y"].iloc[-1],
               c="k", marker="s", s=80, zorder=3, label="end")
    ax.scatter(fall[p + "pose.position.x"], fall[p + "pose.position.y"],
               c="r", marker="X", s=200, zorder=4, label=f"free fall ({fall['t_s']:.2f} s)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (world)")
    ax.set_ylabel("y [m] (world)")
    ax.set_title("spot-07 patrol, Yard B: top-down odometry")
    ax.grid()
    ax.legend()
    fig.savefig("./plots/odom_trajectory.png")
    plt.close(fig)

    # Height over time
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.plot("t_s", p + "pose.position.z", data=df)
    for ts in (start_ns, end_ns):
        ax.axvline((ts - t0) / 1e9, color="r")
    ax.set_xlabel("time since sequence start [s]")
    ax.set_ylabel("body z [m] (world)")
    ax.set_title("Body height from odometry (red: free-fall window)")
    ax.grid()
    fig.savefig("./plots/odom_height.png")
    plt.close(fig)

    return df


def get_tf(timestamps):
    """
    /tf (FrameTransform): each message holds a list of transforms.
    Extract world->body (the robot pose) and cross-check it against odometry,
    and print the static body->front_cam_optical mount.
    """
    start_ns = timestamps[0]
    col = "/tf.FrameTransform.transforms"

    with MosaicoClient.connect("localhost", 6726) as client:
        seq_handler = client.sequence_handler(SEQUENCE)
        if not seq_handler:
            return None
        df = get_topic_df(seq_handler, "/tf")

    def find(transforms, parent, child):
        for tr in transforms:
            if tr["header"]["frame_id"] == parent and tr["target_frame_id"] == child:
                return tr
        return None

    frames = {(tr["header"]["frame_id"], tr["target_frame_id"]) for tr in df[col].iloc[0]}
    print(f"[tf] frames per message: {len(frames)}: {sorted(frames)}")

    r = nearest_row(df, start_ns)
    wb = find(r[col], "world", "body")
    t, q = wb["translation"], wb["rotation"]
    print(f"[tf] world->body at fall start (t={int(r['timestamp_ns'])}): "
          f"x={t['x']:.3f} y={t['y']:.3f} z={t['z']:.3f} m  "
          f"yaw={quat_to_yaw_deg(q['x'], q['y'], q['z'], q['w']):.1f} deg")

    cam = find(r[col], "body", "front_cam_optical")
    if cam:
        ct = cam["translation"]
        print(f"[tf] body->front_cam_optical: x={ct['x']:.3f} y={ct['y']:.3f} z={ct['z']:.3f} m")
    return df


def get_status_log(timestamps, window_s=10):
    """Print the mission log (/robot/status), flagging messages within +-window_s of the fall."""
    start_ns = timestamps[0]
    with MosaicoClient.connect("localhost", 6726) as client:
        seq_handler = client.sequence_handler(SEQUENCE)
        if not seq_handler:
            return None
        df = get_topic_df(seq_handler, "/robot/status")
        t0 = seq_handler.timestamp_ns_min

    for _, r in df.iterrows():
        dt = (r["timestamp_ns"] - start_ns) / 1e9
        flag = ">>" if abs(dt) <= window_s else "  "
        print(f"{flag} {(r['timestamp_ns'] - t0) / 1e9:7.2f} s ({dt:+7.2f} s from fall)  "
              f"{r['/robot/status.String.data']}")
    return df
