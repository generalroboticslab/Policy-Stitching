import socket
import struct
from time import time, sleep
import numpy as np
import threading
import sys
import signal
import os
import math


# Connect to UR5 robot on a real platform
class UR5(object):
    def __init__(self, j_acc=0.1, j_vel=0.1, tool_offset=[0, 0, 0.25, 0, 0, 0]):
        # Set default robot joint acceleration (rad/s^2) and joint velocity (rad/s)
        self.__j_acc = j_acc
        self.__j_vel = j_vel

        # Connect to robot
        robot_ip_key = "ROBOT_IP"
        if robot_ip_key not in os.environ:
            print("Environment variable ROBOT_IP is not found.")
            raise ValueError("ROBOT_IP not found")
        self.__tcp_ip = os.environ[robot_ip_key]
        self.__tcp_port=30002
        self.__rtc_port=30003
        self.__tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__tcp_sock.connect((self.__tcp_ip, self.__tcp_port))
        self.__rtc_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__rtc_sock.connect((self.__tcp_ip, self.__rtc_port))

        # Tool offset for gripper
        self.tool_offset = tool_offset
        tcp_msg = 'set_tcp(p[%f,%f,%f,%f,%f,%f])\n' % tuple(self.tool_offset)
        self.__tcp_sock.send(str.encode(tcp_msg))

        self._suction_seal_threshold = 2.5

        # Set home joint configuration
        # self.__home_j_config = np.asarray(
        #     [-180.0, -90.0, 90.0, -90.0, -90.0, 0.0]
        # ) * np.pi/180.0
        self.__home_j_config = np.asarray(
            [0, -1.1, 1.3, -1.9, -1.571, 0.0]
        )

        # Set joint position and tool pose tolerance (epsilon) for blocking calls
        self.__j_pos_eps = 0.1 * 2  # joints 0.01
        self.__tool_pose_eps = [0.002, 0.002,
                                0.002, 0.01, 0.01, 0.01]  # tool pose

        # Define Denavit-Hartenberg parameters for UR5
        self._ur5_kinematics_d = np.array(
            [0.089159, 0., 0., 0.10915, 0.09465, 0.0823])
        self._ur5_kinematics_a = np.array([0., -0.42500, -0.39225, 0., 0., 0.])

        # Start thread to stream robot state data at 25Hz
        self.__state_data = None
        state_thread = threading.Thread(target=self.get_state_data)
        state_thread.daemon = True
        state_thread.start()
        while self.__state_data is None:
            sleep(0.01)

        # Start second thread to stream robot state data at 125Hz (real-time client)
        self.__rtc_state_data = None
        rtc_state_thread = threading.Thread(target=self.get_rtc_state_data)
        rtc_state_thread.daemon = True
        rtc_state_thread.start()
        while self.__rtc_state_data is None:
            sleep(0.01)

        # Adding signal handlers to stop robot in case following interrupts occur while
        # the robot is still in motion
        print("UR5: registering signal handlers for ctrl+c, ctrl+z")
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    # Get TCP message describing robot state (from primary client)
    def get_state_data(self):
        while True:
            max_tcp_msg_size = 2048
            since = time()
            while True:
                if time() - since < 3:
                    message_size_bytes = bytearray(self.__tcp_sock.recv(4))
                    if len(message_size_bytes) < 4:
                        # Unpacking into int requires 4 bytes
                        continue
                    message_size = struct.unpack("!i", message_size_bytes)[0]
                    # This is hacky but it can work for multiple versions
                    if message_size <= 55 or message_size >= max_tcp_msg_size:
                        continue
                    else:
                        state_data = self.__tcp_sock.recv(message_size-4)
                    if message_size < max_tcp_msg_size and message_size-4 == len(state_data):
                        self.__state_data = state_data
                        break
                else:
                    print(
                        'Timeout: retrieving TCP message exceeded 3 seconds. Restarting connection.')
                    self.__rtc_sock = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    self.__rtc_sock.connect((self.__tcp_ip, self.__rtc_port))
                    break
            sleep(0.01)

    # Get TCP message describing robot state (from real-time client)
    def get_rtc_state_data(self):
        while True:
            max_tcp_msg_size = 2048
            since = time()
            while True:
                if (time()-since) < 3:
                    message_size_bytes = bytearray(self.__rtc_sock.recv(4))
                    message_size = struct.unpack("!i", message_size_bytes)[0]
                    if message_size <= 0:
                        continue
                    else:
                        rtc_state_data = self.__rtc_sock.recv(message_size-4)
                    if message_size < max_tcp_msg_size and message_size-4 == len(rtc_state_data):
                        break
                else:
                    print(
                        'Timeout: retrieving TCP RTC message exceeded 3 seconds. Restarting connection.')
                    self.__tcp_sock = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    self.__tcp_sock.connect((self.__tcp_ip, self.__tcp_port))
            self.__rtc_state_data = rtc_state_data
            sleep(0.005)

    # Parse TCP message describing robot state (from primary client)
    def parse_state_data(self, state_data, req_info):
        # Helper function to skip to specific package byte index in TCP message
        def skip_to_package_index(state_data, pkg_type):
            _ = struct.unpack('!B', state_data[0: 1])[0]
            byte_index = 1
            while byte_index < len(state_data):
                package_size = struct.unpack(
                    "!i", state_data[byte_index:(byte_index+4)])[0]
                byte_index += 4
                package_index = int(struct.unpack(
                    '!B', state_data[(byte_index+0):(byte_index+1)])[0])
                if package_index == pkg_type:
                    byte_index += 1
                    break
                byte_index += package_size - 4
            return byte_index

        # Define functions to parse TCP message for each type of requested information
        def parse_timestamp(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=0)
            timestamp = struct.unpack(
                '!Q', state_data[(byte_index+0):(byte_index+8)])[0]
            return timestamp

        def parse_actual_j_pos(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)
            actual_j_pos = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_pos[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 41
            return actual_j_pos

        def parse_actual_j_vel(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)+16
            actual_j_vel = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_vel[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 41
            return actual_j_vel

        def parse_actual_j_currents(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=1)+24
            actual_j_currents = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_currents[i] = struct.unpack(
                    '!f', state_data[(byte_index+0):(byte_index+4)])[0]
                byte_index += 41
            return actual_j_currents

        def parse_actual_tool_pose(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=4)
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_tool_pose[i] = struct.unpack(
                    '!d', state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_tool_pose

        def parse_tool_analog_input2(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=2)+2
            tool_analog_input2 = struct.unpack(
                '!d', state_data[(byte_index+0):(byte_index+8)])[0]
            return tool_analog_input2

        def parse_analog_input1(state_data):
            byte_index = skip_to_package_index(state_data, pkg_type=3)+14
            analog_input1 = struct.unpack(
                '!d', state_data[(byte_index+0):(byte_index+8)])[0]
            return analog_input1

        # Map requested info to parsing function and sub-package type
        parse_func = {
            'timestamp': parse_timestamp,
            'actual_j_pos': parse_actual_j_pos,
            'actual_j_vel': parse_actual_j_vel,
            'actual_j_currents': parse_actual_j_currents,
            'actual_tool_pose': parse_actual_tool_pose,
            'tool_analog_input2': parse_tool_analog_input2,
            'analog_input1': parse_analog_input1
        }
        return parse_func[req_info](state_data)

    # Parse TCP message describing robot state (from real-time client)
    def parse_rtc_state_data(self, rtc_state_data, req_info):
        # Define functions to parse TCP message for each type of requested information
        def parse_timestamp(rtc_state_data):
            byte_index = 0
            timestamp = struct.unpack(
                '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
            return timestamp

        def parse_actual_j_pos(rtc_state_data):
            byte_index = 8+48*5
            actual_j_pos = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_pos[i] = struct.unpack(
                    '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_j_pos

        def parse_actual_j_vel(rtc_state_data):
            byte_index = 8+48*6
            actual_j_vel = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_vel[i] = struct.unpack(
                    '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_j_vel

        def parse_actual_j_currents(rtc_state_data):
            byte_index = 8+48*7
            actual_j_currents = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_j_currents[i] = struct.unpack(
                    '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_j_currents

        def parse_actual_tool_pose(rtc_state_data):
            byte_index = 8+48*8+24+120+48
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_tool_pose[i] = struct.unpack(
                    '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_tool_pose

        def parse_actual_tool_vel(rtc_state_data):
            byte_index = 8+48*8+24+120+48*2
            actual_tool_vel = [0, 0, 0, 0, 0, 0]
            for i in range(6):
                actual_tool_vel[i] = struct.unpack(
                    '!d', rtc_state_data[(byte_index+0):(byte_index+8)])[0]
                byte_index += 8
            return actual_tool_vel

        # Map requested info to parsing function and sub-package type
        parse_func = {
            'timestamp': parse_timestamp,
            'actual_j_pos': parse_actual_j_pos,
            'actual_j_vel': parse_actual_j_vel,
            'actual_j_currents': parse_actual_j_currents,
            'actual_tool_pose': parse_actual_tool_pose,
            'actual_tool_vel': parse_actual_tool_vel
        }
        return parse_func[req_info](rtc_state_data)

    # Move joints to specified positions or move tool to specified pose
    def movej(self, use_pos, params, blocking=False, j_acc=-1, j_vel=-1):
        # Apply default joint speeds
        if j_acc == -1:
            j_acc = self.__j_acc
        if j_vel == -1:
            j_vel = self.__j_vel

        # Move robot
        tcp_msg = "def process():\n"
        tcp_msg += f" stopj({j_acc})\n"
        tcp_msg += f" movej({'p' if use_pos else ''}[{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{params[5]}],a={j_acc},v={j_vel},t=0.0,r=0.0)\n"
        tcp_msg += "end\n"
        self.__tcp_sock.send(str.encode(tcp_msg))
        sleep(0.5)  # 1.0

        # If blocking call, pause until robot stops moving
        if blocking:
            for _ in range(100):
                state_data = self.__state_data
                actual_j_pos = self.parse_state_data(state_data, 'actual_j_pos')
                actual_j_vel = self.parse_state_data(state_data, 'actual_j_vel')
                actual_tool_pose = self.parse_state_data(
                    state_data, 'actual_tool_pose')
                # Handle repeat axis angle rotations
                actual_tool_pose_mirror = np.asarray(list(actual_tool_pose))
                actual_tool_pose_mirror[3:6] = -actual_tool_pose_mirror[3:6]
                if use_pos:
                    if (
                            all([np.abs(actual_tool_pose[i] - params[i]) < self.__tool_pose_eps[i] for i in
                                 range(6)]) or
                            all([np.abs(actual_tool_pose_mirror[i] - params[i]) < self.__tool_pose_eps[i] for i in
                                 range(6)])
                    ) and np.sum(actual_j_vel) < 10:
                        break
                else:
                    if all([np.abs(actual_j_pos[i] - params[i]) < self.__j_pos_eps for i in range(6)]) and np.sum(
                            actual_j_vel) < 10 * 2:  # originally 10
                        break

                self.homej(blocking=False)
                print("cannot move to that pos, back to home pos")

        # while blocking:
        #     state_data = self.__state_data
        #     actual_j_pos = self.parse_state_data(state_data, 'actual_j_pos')
        #     actual_j_vel = self.parse_state_data(state_data, 'actual_j_vel')
        #     actual_tool_pose = self.parse_state_data(
        #         state_data, 'actual_tool_pose')
        #     # Handle repeat axis angle rotations
        #     actual_tool_pose_mirror = np.asarray(list(actual_tool_pose))
        #     actual_tool_pose_mirror[3:6] = -actual_tool_pose_mirror[3:6]
        #     if use_pos:
        #         if (
        #             all([np.abs(actual_tool_pose[i]-params[i]) < self.__tool_pose_eps[i] for i in range(6)]) or
        #             all([np.abs(actual_tool_pose_mirror[i]-params[i]) < self.__tool_pose_eps[i] for i in range(6)])
        #         ) and np.sum(actual_j_vel) < 10:
        #             break
        #     else:
        #         if all([np.abs(actual_j_pos[i]-params[i]) < self.__j_pos_eps for i in range(6)]) and np.sum(actual_j_vel) < 10*2:  # originally 10
        #             break

    # Move joints to home joint configuration
    def homej(self, blocking=False):
        self.movej(use_pos=False, params=self.__home_j_config, blocking=blocking)

    # Move tool in a straight line
    def movep(self, params, blocking=False):
        # Move robot
        tcp_msg = "def process():\n"
        tcp_msg += f" stopj({self.__j_acc})\n"
        tcp_msg += f" movep(p[{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{params[5]}],a={self.__j_acc},v={self.__j_vel},r=0.0)\n"
        tcp_msg += "end\n"
        self.__tcp_sock.send(str.encode(tcp_msg))

        # If blocking call, pause until robot stops moving
        while blocking:
            state_data = self.__state_data
            actual_j_pos = self.parse_state_data(
                state_data, 'actual_j_pos')
            actual_j_vel = self.parse_state_data(
                state_data, 'actual_j_vel')
            actual_tool_pose = self.parse_state_data(
                state_data, 'actual_tool_pose')
            # Handle repeat axis angle rotations
            actual_tool_pose_mirror = np.asarray(list(actual_tool_pose))
            actual_tool_pose_mirror[3:6] = -actual_tool_pose_mirror[3:6]
            if (
                all([np.abs(actual_tool_pose[i]-params[i]) < self.__tool_pose_eps[i] for i in range(6)]) or
                all([np.abs(actual_tool_pose_mirror[i]-params[i]) < self.__tool_pose_eps[i] for i in range(6)])
            ) and np.sum(actual_j_vel) < 0.01:
                break

    # Move robot with specified joint velocities
    def speedj(self, params, timeout):
        tcp_msg = "def process():\n"
        tcp_msg += f" speedj([{params[0]},{params[1]},{params[2]},{params[3]},{params[4]},{params[5]}],a={self.__j_acc},t_min={timeout})\n"
        tcp_msg += " stopj(8.0)\n"
        tcp_msg += "end\n"
        self.__tcp_sock.send(str.encode(tcp_msg))

    def close_gripper(self, blocking=False):
        tcp_msg = "set_digital_out(8,True)\n"
        self.__tcp_sock.send(str.encode(tcp_msg))
        if blocking:
            sleep(0.5)
        return True  # gripper_closed

    def open_gripper(self, blocking=False):
        tcp_msg = "set_digital_out(8,False)\n"
        self.__tcp_sock.send(str.encode(tcp_msg))
        if blocking:
            sleep(0.5)

    # Check if something is in between gripper fingers by measuring grasp width
    def check_grasp(self):
        state_data = self.__state_data
        analog_input1 = self.parse_state_data(state_data, 'analog_input1')

        # Find peak in analog input
        timeout_t0 = time()
        while True:
            state_data = self.__state_data
            new_analog_input1 = self.parse_state_data(
                state_data, 'analog_input1')
            timeout_t1 = time()
            if (
                new_analog_input1 > 2.0 and
                abs(new_analog_input1 - analog_input1) > 0.0 and
                abs(new_analog_input1 - analog_input1) < 0.1
            ) or timeout_t1 - timeout_t0 > 5:
                print(analog_input1)
                return analog_input1 > self._suction_seal_threshold
            analog_input1 = new_analog_input1

    # Get current 6D pose of end effector by reading robot state data

    def get_tool_pose(self):
        state_data = self.__state_data
        tool_pose = self.parse_state_data(state_data, 'actual_tool_pose')
        tool_trans = np.asarray(tool_pose[:3]).reshape(3, 1)
        tool_rotm = angle2rotm(urx2angle(tool_pose[3:]))[:3, :3]
        tool2robot = np.concatenate((np.concatenate(
            (tool_rotm, tool_trans), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
        return tool2robot

    # Get current joint configuration by reading robot state data
    def get_j_config(self):
        state_data = self.__state_data
        actual_j_pos = self.parse_state_data(state_data, 'actual_j_pos')
        return actual_j_pos

    # Block until robot stops moving
    def block_until_stop(self):
        while True:
            actual_j_pos = self.parse_state_data(
                self.__state_data, 'actual_j_pos')
            actual_j_vel = self.parse_state_data(
                self.__state_data, 'actual_j_vel')
            if all([np.abs(actual_j_pos[i] - self.__home_j_config[i]) < self.__j_pos_eps for i in range(6)]) and\
                np.sum(actual_j_vel) < 0.01:
                break
            sleep(0.01)

    def get_tool_position_forward_kinematics(self, joint_angles):
        d_with_tool = self._ur5_kinematics_d.copy()
        d_with_tool[5] += self.tool_offset[2]
        return forward_kinematics_6dof(d_with_tool, self._ur5_kinematics_a, joint_angles)

    def signal_handler(self, sig, frame):
        # Send stop joints signal to robot in case of interrupt signals and gracefully exit
        tcp_msg = 'def process():\n'
        tcp_msg += ' stopj(%f)\n' % (self.__j_acc)
        tcp_msg += 'end\n'
        self.__tcp_sock.send(str.encode(tcp_msg))
        sys.exit(0)


def angle2rotm(angle_axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke
    angle = angle_axis[0]
    axis = angle_axis[1:]

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M[:3,:3]

# Convert from URx rotation format to axis angle
def urx2angle(v):
    angle = np.linalg.norm(v)
    axis = v/angle
    return np.insert(axis,0,angle)

# Use forward kinematics to get position of end effector using devanit-hartenberge for 6DOF
def forward_kinematics_6dof(d,a,joint_angles):
    px = -(d[4]*(np.sin(joint_angles[0])*np.cos(joint_angles[1] + joint_angles[2] + joint_angles[3])-np.cos(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])))/2.0+(d[4]*(np.sin(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])+np.cos(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])))/2.0+d[3]*np.sin(joint_angles[0])-(d[5]*(np.cos(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])-np.sin(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3]))*np.sin(joint_angles[4]))/2.0-(d[5]*(np.cos(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])+np.sin(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3]))*np.sin(joint_angles[4]))/2.0 + a[1]*np.cos(joint_angles[0])*np.cos(joint_angles[1]) + d[5]*np.cos(joint_angles[4])*np.sin(joint_angles[0]) + a[2]*np.cos(joint_angles[0])*np.cos(joint_angles[1])*np.cos(joint_angles[2]) - a[2]*np.cos(joint_angles[0])*np.sin(joint_angles[1])*np.sin(joint_angles[2])
    py = -(d[4]*(np.cos(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])-np.sin(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])))/2.0+(d[4]*(np.cos(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])+np.sin(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])))/2.0-d[3]*np.cos(joint_angles[0])-(d[5]*(np.sin(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])+np.cos(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3]))*np.sin(joint_angles[4]))/2.0-(d[5]*(np.sin(joint_angles[0])*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])-np.cos(joint_angles[0])*np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3]))*np.sin(joint_angles[4]))/2.0 - d[5]*np.cos(joint_angles[0])*np.cos(joint_angles[4]) + a[1]*np.cos(joint_angles[1])*np.sin(joint_angles[0]) + a[2]*np.cos(joint_angles[1])*np.cos(joint_angles[2])*np.sin(joint_angles[0]) - a[2]*np.sin(joint_angles[0])*np.sin(joint_angles[1])*np.sin(joint_angles[2])
    pz = d[0] + (d[5]*(np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])*np.cos(joint_angles[4]) - np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])*np.sin(joint_angles[4])))/2.0 + a[2]*(np.sin(joint_angles[1])*np.cos(joint_angles[2]) + np.cos(joint_angles[1])*np.sin(joint_angles[2])) + a[1]*np.sin(joint_angles[1]) - (d[5]*(np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])*np.cos(joint_angles[4]) + np.sin(joint_angles[1]+joint_angles[2]+joint_angles[3])*np.sin(joint_angles[4])))/2.0 - d[4]*np.cos(joint_angles[1]+joint_angles[2]+joint_angles[3])
    return np.array([px,py,pz])
