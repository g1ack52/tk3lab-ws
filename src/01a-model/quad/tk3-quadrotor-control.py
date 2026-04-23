#
import genomix
import os
import time
import numpy as np
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ws_dir = os.environ['TK3LAB_WS']
log_dir = os.path.join(ws_dir, 'logs', '01a-model', 'quad')
output_dir = os.path.join(ws_dir, "outputs", "01a-model", "quad")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# this connects to components running on the same host (localhost)
g = genomix.connect()
# to instead control components running on the remote computer "hostname" use
# g = genomix.connect('hostname')

# adapt path to your setup
g.rpath(os.environ['HOME'] + '/openrobots/lib/genom/pocolibs/plugins')

# load components clients
optitrack = g.load('optitrack')
rotorcraft = g.load('rotorcraft')
pom = g.load('pom')
nhfc = g.load('nhfc')

# --- setup ----------------------------------------------------------------
#
# configure components, to be called interactively
# VIENE COSTRUITA LA CATENA DI SIMULAZIONE DEL QUAD
def setup():
  # optitrack
  #
  # connect to the simulated optitrack system on localhost::1509
  optitrack.connect({
    'host': 'localhost', 'host_port': '1509', 'mcast': '', 'mcast_port': '0'
  })


  # rotorcraft
  #
  # connect to the simulated quadrotor VIA SERIAL
  rotorcraft.connect({'serial': '/tmp/pty-qr4', 'baud': 0})
  #qr4 É L'ISTANZA DEL QUAD DEFINITA NEL .world

  # get IMU at 1kHz and motor data at 20Hz
  rotorcraft.set_sensor_rate({'rate': {
    'imu': 1000, 'mag': 0, 'motor': 20, 'battery': 1
  }})

  # Filter IMU: 20Hz cut-off frequency for gyroscopes and 5Hz for
  # accelerometers. This is important for cancelling vibrations.
  rotorcraft.set_imu_filter({
    'gfc': [20, 20, 20], 'afc': [5, 5, 5], 'mfc': [20, 20, 20]
  })

  # read propellers velocities from nhfc controller
  rotorcraft.connect_port({
    'local': 'rotor_input', 'remote': 'nhfc/rotor_input'
  })


  # nhfc
  #
  # configure quadrotor geometry: 4 rotors, not tilted, 23cm arms
  nhfc.set_gtmrp_geom({
    'rotors': 4, 'cx': 0, 'cy': 0, 'cz': 0, 'armlen': 0.23, 'mass': 1.28,
    'rx':0, 'ry': 0, 'rz': -1, 'cf': 6.5e-4, 'ct': 1e-5
  })

  # emergency descent parameters
  nhfc.set_emerg({'emerg': {
    'descent': 0.1, 'dx': 0.5, 'dq': 1, 'dv': 3, 'dw': 3
  }})

  # PID tuning
  nhfc.set_saturation({'sat': {'x': 1, 'v': 1, 'ix': 0}})
  nhfc.set_servo_gain({ 'gain': {
    'Kpxy': 5, 'Kpz': 5, 'Kqxy': 4, 'Kqz': 0.1,
    'Kvxy': 6, 'Kvz': 6, 'Kwxy': 1, 'Kwz': 0.1,
    'Kixy': 0, 'Kiz': 0
  }})

  # use tilt-prioritized controller
  nhfc.set_control_mode({'att_mode': '::nhfc::tilt_prioritized'})

  # read measured propeller velocities from rotorcraft
  nhfc.connect_port({
    'local': 'rotor_measure', 'remote': 'rotorcraft/rotor_measure'
  })

  # read current state from pom
  nhfc.connect_port({
    'local': 'state', 'remote': 'pom/frame/robot'
  })


  # pom
  #
  # configure kalman filter
  pom.set_prediction_model('::pom::constant_acceleration')
  pom.set_process_noise({'max_jerk': 100, 'max_dw': 50})

  # allow sensor data up to 250ms old
  pom.set_history_length({'history_length': 0.25})

  # configure magnetic field
  pom.set_mag_field({'magdir': {
    'x': 23.8e-06, 'y': -0.4e-06, 'z': -39.8e-06
  }})

  # read IMU and magnetometers from rotorcraft
  pom.connect_port({'local': 'measure/imu', 'remote': 'rotorcraft/imu'})
  pom.add_measurement('imu')
  pom.connect_port({'local': 'measure/mag', 'remote': 'rotorcraft/mag'})
  pom.add_measurement('mag')

  # read position and orientation from optitrack
  pom.connect_port({
    'local': 'measure/mocap', 'remote': 'optitrack/bodies/QR_4'
  })
  pom.add_measurement('mocap')


# --- start ----------------------------------------------------------------
#
# Spin the motors and servo on current position. To be called interactively
# QUA VENGONO AVVIATI I LOG!!!!
def start():
  pom.log_state(os.path.join(log_dir, 'pom.log'))
  pom.log_measurements(os.path.join(log_dir, 'pom-measurements.log'))

  optitrack.set_logfile(os.path.join(log_dir, 'opti.log'))

  rotorcraft.log(os.path.join(log_dir, 'rotorcraft.log'))
  rotorcraft.start()
  rotorcraft.servo(ack=True) # this runs until stopped or input error

  nhfc.log(os.path.join(log_dir, 'nhfc.log'))
  nhfc.set_current_position() # hover on current position


# --- stop -----------------------------------------------------------------
#
# Stop motors. To be called interactively
def stop():
  rotorcraft.stop()
  rotorcraft.log_stop()

  nhfc.stop()
  nhfc.log_stop()

  pom.log_stop()

  optitrack.unset_logfile()



# assignment 1 ######################################################################

# --- simulation -----------------------------------------------------------------
#
def simulation():
  setup()
  start()

  time.sleep(2)
  nhfc.set_position(1, 1, 1, 0)

  time.sleep(5)
  nhfc.set_position(1, -1, 1, math.pi/2)

  time.sleep(5)
  nhfc.set_position(0, 0, 0, 0)

  time.sleep(5)
  stop()



# --- graph -----------------------------------------------------------------
#
def read_genom_log(path):
    """
    Legge un log GenoM3:
    - ignora le righe commentate (#)
    - usa la prima riga non commentata come header
    - converte tutto ciò che può in numerico
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    data_lines = [line for line in lines if not line.startswith("#")]
    if not data_lines:
        raise ValueError(f"Nessun dato trovato in {path}")

    header = data_lines[0].split()
    rows = [line.split() for line in data_lines[1:]]

    df = pd.DataFrame(rows, columns=header)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    return df

def data():
  pom = read_genom_log(os.path.join(log_dir, "pom.log"))
  nhfc = read_genom_log(os.path.join(log_dir, "nhfc.log"))
  rot = read_genom_log(os.path.join(log_dir, "rotorcraft.log"))
  opti = read_genom_log(os.path.join(log_dir, "opti.log"))

  t0 = min(
      pom["ts"].min(),
      nhfc["ts"].min(),
      rot["ts"].min(),
      opti["ts"].min()
  )

  pom["t"] = pom["ts"] - t0
  nhfc["t"] = nhfc["ts"] - t0
  rot["t"] = rot["ts"] - t0
  opti["t"] = opti["ts"] - t0

  pom["roll_deg"] = np.degrees(pom["roll"])
  pom["pitch_deg"] = np.degrees(pom["pitch"])
  pom["yaw_deg"] = np.degrees(pom["yaw"])

  pom["wx_deg_s"] = np.degrees(pom["wx"])
  pom["wy_deg_s"] = np.degrees(pom["wy"])
  pom["wz_deg_s"] = np.degrees(pom["wz"])

  nhfc["rolld_deg"] = np.degrees(nhfc["rolld"])
  nhfc["pitchd_deg"] = np.degrees(nhfc["pitchd"])
  nhfc["yawd_deg"] = np.degrees(nhfc["yawd"])

  nhfc["e_rx_deg"] = np.degrees(nhfc["e_rx"])
  nhfc["e_ry_deg"] = np.degrees(nhfc["e_ry"])
  nhfc["e_rz_deg"] = np.degrees(nhfc["e_rz"])

  nhfc["wxd_deg_s"] = np.degrees(nhfc["wxd"])
  nhfc["wyd_deg_s"] = np.degrees(nhfc["wyd"])
  nhfc["wzd_deg_s"] = np.degrees(nhfc["wzd"])

  nhfc["e_wx_deg_s"] = np.degrees(nhfc["e_wx"])
  nhfc["e_wy_deg_s"] = np.degrees(nhfc["e_wy"])
  nhfc["e_wz_deg_s"] = np.degrees(nhfc["e_wz"])

  return pom, nhfc, rot, opti

def plot_position_tracking(pom, nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  # x
  axs[0].plot(pom["t"], pom["x"], label="x measured")
  axs[0].plot(nhfc["t"], nhfc["xd"], "--", label="x desired")
  axs[0].set_ylabel("x [m]")
  axs[0].set_title("Position tracking")
  axs[0].grid(True)
  axs[0].legend()

  # y
  axs[1].plot(pom["t"], pom["y"], label="y measured")
  axs[1].plot(nhfc["t"], nhfc["yd"], "--", label="y desired")
  axs[1].set_ylabel("y [m]")
  axs[1].grid(True)
  axs[1].legend()

  # z
  axs[2].plot(pom["t"], pom["z"], label="z measured")
  axs[2].plot(nhfc["t"], nhfc["zd"], "--", label="z desired")
  axs[2].set_ylabel("z [m]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_position_tracking.png"))
  #plt.show()
  plt.close(fig)


def plot_attitude_tracking(pom, nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  # roll
  axs[0].plot(pom["t"], pom["roll_deg"], label="roll measured")
  axs[0].plot(nhfc["t"], nhfc["rolld_deg"], "--", label="roll desired")
  axs[0].set_ylabel("roll [deg]")
  axs[0].set_title("Attitude tracking")
  axs[0].grid(True)
  axs[0].legend()

  # pitch
  axs[1].plot(pom["t"], pom["pitch_deg"], label="pitch measured")
  axs[1].plot(nhfc["t"], nhfc["pitchd_deg"], "--", label="pitch desired")
  axs[1].set_ylabel("pitch [deg]")
  axs[1].grid(True)
  axs[1].legend()

  # yaw
  axs[2].plot(pom["t"], pom["yaw_deg"], label="yaw measured")
  axs[2].plot(nhfc["t"], nhfc["yawd_deg"], "--", label="yaw desired")
  axs[2].set_ylabel("yaw [deg]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_attitude_tracking.png"))
  #plt.show()
  plt.close(fig)


def plot_linear_velocity_tracking(pom, nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  # vx
  axs[0].plot(pom["t"], pom["vx"], label="vx measured")
  axs[0].plot(nhfc["t"], nhfc["vxd"], "--", label="vx desired")
  axs[0].set_ylabel("vx [m/s]")
  axs[0].set_title("Linear velocity tracking")
  axs[0].grid(True)
  axs[0].legend()

  # vy
  axs[1].plot(pom["t"], pom["vy"], label="vy measured")
  axs[1].plot(nhfc["t"], nhfc["vyd"], "--", label="vy desired")
  axs[1].set_ylabel("vy [m/s]")
  axs[1].grid(True)
  axs[1].legend()

  # vz
  axs[2].plot(pom["t"], pom["vz"], label="vz measured")
  axs[2].plot(nhfc["t"], nhfc["vzd"], "--", label="vz desired")
  axs[2].set_ylabel("vz [m/s]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_linear_velocity_tracking.png"))
  #plt.show()
  plt.close(fig)


def plot_angular_velocity_tracking(pom, nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  # wx
  axs[0].plot(pom["t"], pom["wx_deg_s"], label="wx measured")
  axs[0].plot(nhfc["t"], nhfc["wxd_deg_s"], "--", label="wx desired")
  axs[0].set_ylabel("wx [deg/s]")
  axs[0].set_title("Angular velocity tracking")
  axs[0].grid(True)
  axs[0].legend()

  # wy
  axs[1].plot(pom["t"], pom["wy_deg_s"], label="wy measured")
  axs[1].plot(nhfc["t"], nhfc["wyd_deg_s"], "--", label="wy desired")
  axs[1].set_ylabel("wy [deg/s]")
  axs[1].grid(True)
  axs[1].legend()

  # wz
  axs[2].plot(pom["t"], pom["wz_deg_s"], label="wz measured")
  axs[2].plot(nhfc["t"], nhfc["wzd_deg_s"], "--", label="wz desired")
  axs[2].set_ylabel("wz [deg/s]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_angular_velocity_tracking.png"))
  #plt.show()
  plt.close(fig)


def plot_linear_acceleration_tracking(pom, nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  # ax
  axs[0].plot(pom["t"], pom["ax"], label="ax measured")
  axs[0].plot(nhfc["t"], nhfc["axd"], "--", label="ax desired")
  axs[0].set_ylabel("ax [m/s^2]")
  axs[0].set_title("Linear acceleration tracking")
  axs[0].grid(True)
  axs[0].legend()

  # ay
  axs[1].plot(pom["t"], pom["ay"], label="ay measured")
  axs[1].plot(nhfc["t"], nhfc["ayd"], "--", label="ay desired")
  axs[1].set_ylabel("ay [m/s^2]")
  axs[1].grid(True)
  axs[1].legend()

  # az
  axs[2].plot(pom["t"], pom["az"], label="az measured")
  axs[2].plot(nhfc["t"], nhfc["azd"], "--", label="az desired")
  axs[2].set_ylabel("az [m/s^2]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_linear_acceleration_tracking.png"))
  #plt.show()
  plt.close(fig)

def plot_position_errors(nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axs[0].plot(nhfc["t"], nhfc["e_x"])
  axs[0].set_ylabel("e_x [m]")
  axs[0].set_title("Position errors")
  axs[0].grid(True)

  axs[1].plot(nhfc["t"], nhfc["e_y"])
  axs[1].set_ylabel("e_y [m]")
  axs[1].grid(True)

  axs[2].plot(nhfc["t"], nhfc["e_z"])
  axs[2].set_ylabel("e_z [m]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_position_errors.png"))
  #plt.show()
  plt.close(fig)

def plot_attitude_errors(nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axs[0].plot(nhfc["t"], nhfc["e_rx_deg"])
  axs[0].set_ylabel("e_roll [deg]")
  axs[0].set_title("Attitude errors")
  axs[0].grid(True)

  axs[1].plot(nhfc["t"], nhfc["e_ry_deg"])
  axs[1].set_ylabel("e_pitch [deg]")
  axs[1].grid(True)

  axs[2].plot(nhfc["t"], nhfc["e_rz_deg"])
  axs[2].set_ylabel("e_yaw [deg]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_attitude_errors.png"))
  #plt.show()
  plt.close(fig)

def plot_linear_velocity_errors(nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axs[0].plot(nhfc["t"], nhfc["e_vx"])
  axs[0].set_ylabel("e_vx [m/s]")
  axs[0].set_title("Linear velocity errors")
  axs[0].grid(True)

  axs[1].plot(nhfc["t"], nhfc["e_vy"])
  axs[1].set_ylabel("e_vy [m/s]")
  axs[1].grid(True)

  axs[2].plot(nhfc["t"], nhfc["e_vz"])
  axs[2].set_ylabel("e_vz [m/s]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_linear_velocity_errors.png"))
  #plt.show()
  plt.close(fig)

def plot_angular_velocity_errors(nhfc):
  fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

  axs[0].plot(nhfc["t"], nhfc["e_wx_deg_s"])
  axs[0].set_ylabel("e_wx [deg/s]")
  axs[0].set_title("Angular velocity errors")
  axs[0].grid(True)

  axs[1].plot(nhfc["t"], nhfc["e_wy_deg_s"])
  axs[1].set_ylabel("e_wy [deg/s]")
  axs[1].grid(True)

  axs[2].plot(nhfc["t"], nhfc["e_wz_deg_s"])
  axs[2].set_ylabel("e_wz [deg/s]")
  axs[2].set_xlabel("time [s]")
  axs[2].grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_angular_velocity_errors.png"))
  #plt.show()
  plt.close(fig)

def plot_control_inputs(nhfc):
  fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

  axs[0, 0].plot(nhfc["t"], nhfc["fx"])
  axs[0, 0].set_ylabel("fx [N]")
  axs[0, 0].set_title("Control forces and torques")
  axs[0, 0].grid(True)

  axs[0, 1].plot(nhfc["t"], nhfc["fy"])
  axs[0, 1].set_ylabel("fy [N]")
  axs[0, 1].grid(True)

  axs[0, 2].plot(nhfc["t"], nhfc["fz"])
  axs[0, 2].set_ylabel("fz [N]")
  axs[0, 2].grid(True)

  axs[1, 0].plot(nhfc["t"], nhfc["tx"])
  axs[1, 0].set_ylabel("tx [Nm]")
  axs[1, 0].set_xlabel("time [s]")
  axs[1, 0].grid(True)

  axs[1, 1].plot(nhfc["t"], nhfc["ty"])
  axs[1, 1].set_ylabel("ty [Nm]")
  axs[1, 1].set_xlabel("time [s]")
  axs[1, 1].grid(True)

  axs[1, 2].plot(nhfc["t"], nhfc["tz"])
  axs[1, 2].set_ylabel("tz [Nm]")
  axs[1, 2].set_xlabel("time [s]")
  axs[1, 2].grid(True)

  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_control_inputs.png"))

  #plt.show()  
  plt.close(fig)

def plot_propeller_speeds(rot, n_props=4):
  fig, axs = plt.subplots(n_props, 1, figsize=(12, 3.5 * n_props), sharex=True)

  if n_props == 1:
    axs = [axs]

  step = 20   # prova 20, 50 o 100

  for i in range(n_props):
    cmd_col = f"cmd_v{i}"
    meas_col = f"meas_v{i}"

    t_plot = rot["t"].iloc[::step]
    cmd_plot = rot[cmd_col].iloc[::step]
    meas_plot = rot[meas_col].iloc[::step]

    axs[i].plot(t_plot, cmd_plot, "--", label=cmd_col, linewidth=1.0)
    axs[i].plot(t_plot, meas_plot, label=meas_col, linewidth=0.8)
    axs[i].set_ylabel("Hz")
    axs[i].set_title(f"Propeller {i}")
    axs[i].grid(True)
    axs[i].legend()

  axs[-1].set_xlabel("time [s]")
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "plot_propeller_speeds.png"))
  plt.close(fig)


def graph():
  pom, nhfc, rot, opti = data()

  print("Saving position tracking...")
  plot_position_tracking(pom, nhfc)

  print("Saving attitude tracking...")
  plot_attitude_tracking(pom, nhfc)

  print("Saving position errors...")
  plot_position_errors(nhfc)

  print("Saving attitude errors...")
  plot_attitude_errors(nhfc)

  print("Saving linear velocity errors...")
  plot_linear_velocity_errors(nhfc)

  print("Saving angular velocity errors...")
  plot_angular_velocity_errors(nhfc)

  print("Saving control inputs...")
  plot_control_inputs(nhfc)

  print("Saving propeller speeds...")
  plot_propeller_speeds(rot, n_props=4)

  print("Done.")