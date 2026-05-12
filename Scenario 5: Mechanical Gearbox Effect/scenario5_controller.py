import subprocess
import threading
import time
import math
import csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed.")
    print("Run: sudo apt update && sudo apt install -y python3-matplotlib")
    raise

# -----------------------------
# Settings
# -----------------------------
TOTAL_TIME = 25.0          # seconds
DT = 0.02                   # sample time (seconds)

CMD_TOPIC = "/model/solar_tracker/joint/tracking_joint/cmd_pos"
STATE_TOPIC = "/world/tracker_world/model/solar_tracker/joint_state"

latest_angle = 0.0
running = True
reader_process = None
angle_lock = threading.Lock()


# -----------------------------
# Reference function
# theta_ref(t) = -cos(pi*t/T)
# -----------------------------
def reference_angle(t):
    return -math.cos(math.pi * t / TOTAL_TIME)


# -----------------------------
# Read actual joint angle from Gazebo
# -----------------------------
def state_reader():
    global latest_angle, running, reader_process

    cmd = ["stdbuf", "-oL", "ign", "topic", "-e", "-t", STATE_TOPIC]
    reader_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    in_axis1 = False
    axis1_depth = 0

    for raw_line in reader_process.stdout:
        if not running:
            break

        line = raw_line.strip()

        # Start reading inside axis1 block
        if line.startswith("axis1 {"):
            in_axis1 = True
            axis1_depth = 1
            continue

        if in_axis1:
            # Read the actual joint position
            if line.startswith("position:"):
                try:
                    value = float(line.split("position:")[1].strip())
                    with angle_lock:
                        latest_angle = value
                except ValueError:
                    pass

            # Track nested braces properly
            axis1_depth += line.count("{")
            axis1_depth -= line.count("}")

            # End of axis1 block
            if axis1_depth <= 0:
                in_axis1 = False
                axis1_depth = 0

    if reader_process.poll() is None:
        reader_process.terminate()

# -----------------------------
# Send desired angle command to Gazebo
# -----------------------------
def publish_angle(angle):
    msg = f"data: {angle}"
    subprocess.run(
        ["ign", "topic", "-t", CMD_TOPIC, "-m", "ignition.msgs.Double", "-p", msg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# -----------------------------
# Safely get latest measured angle
# -----------------------------
def get_latest_angle():
    with angle_lock:
        return latest_angle


# -----------------------------
# Main
# -----------------------------
def main():
    global running, reader_process

    print("Starting controller...")
    print("Make sure Gazebo is already open and the simulation is running.")
    print("The script will run for 25 seconds.")

    # Start background reader
    thread = threading.Thread(target=state_reader, daemon=True)
    thread.start()

    # Give the reader a moment to connect
    time.sleep(1.0)

    times = []
    refs = []
    actuals = []
    errors = []

    start_time = time.time()
    next_time = start_time

    try:
        while True:
            now = time.time()
            t = now - start_time

            if t > TOTAL_TIME:
                break

            theta_ref = reference_angle(t)
            publish_angle(theta_ref)

            theta_actual = get_latest_angle()
            error = theta_ref - theta_actual

            times.append(t)
            refs.append(theta_ref)
            actuals.append(theta_actual)
            errors.append(error)

            next_time += DT
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        running = False
        if reader_process is not None and reader_process.poll() is None:
            reader_process.terminate()

    # -----------------------------
    # Error metrics
    # -----------------------------
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))

    if len(times) >= 2:
        avg_dt = sum(t2 - t1 for t1, t2 in zip(times[:-1], times[1:])) / (len(times) - 1)
    else:
        avg_dt = 0.0

    print("\nFinished.")
    print(f"Samples collected = {len(times)}")
    print(f"Average dt        = {avg_dt:.6f} s")
    print(f"MAE  = {mae:.6f} rad")
    print(f"RMSE = {rmse:.6f} rad")

    # -----------------------------
    # Save CSV data
    # -----------------------------
    with open("tracking_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "theta_ref_rad", "theta_actual_rad", "error_rad"])
        for t, r, a, e in zip(times, refs, actuals, errors):
            writer.writerow([t, r, a, e])

    # -----------------------------
    # Plot 1: reference vs actual
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(times, refs, label="Target (Sun)")
    plt.plot(times, actuals, label="Actual (Panel)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Solar Panel Tracking Performance (10:1 Gearbox)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tracking_plot.png", dpi=200)

    # -----------------------------
    # Plot 2: error
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(times, errors)
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.title("Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_plot.png", dpi=200)

    print("Saved files:")
    print("  tracking_data.csv")
    print("  tracking_plot.png")
    print("  error_plot.png")


if __name__ == "__main__":
    main()