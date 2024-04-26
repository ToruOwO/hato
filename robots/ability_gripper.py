import platform
import struct
import threading
import time

import numpy as np
import serial
from serial.tools import list_ports


## Send Miscellanous Command to Ability Hand
def create_misc_msg(cmd):
    barr = []
    barr.append((struct.pack("<B", 0x50))[0])  # device ID
    barr.append((struct.pack("<B", cmd))[0])  # command!
    sum = 0
    for b in barr:
        sum = sum + b
    chksum = (-sum) & 0xFF
    barr.append(chksum)
    return barr


def call_gripper(gripper, cmd, end_event, debug):
    i = 0
    while not end_event.is_set():
        gripper._serial_comm(cmd, debug=debug)
        time.sleep(0.01)
        # print("call ", i, end_event.is_set())
        i += 1


class AbilityGripper:
    # default reply mode 0x11 stands for position mode variant 2
    def __init__(
        self,
        baud_rate=460800,
        hand_addr=0x50,
        reply_mode=0x11,
        port_idx=-1,
        grip_range=110,
    ):
        self.port_idx = port_idx
        self.baud_rate = baud_rate
        self.hand_addr = hand_addr
        self.reply_mode = reply_mode
        self.ser = []
        self.reset_count = 0
        self.total_count = 0

        # recommended by Jesse from Psyonic
        print("AbilityGripper: grip range", grip_range)
        self.upper_ranges = [int(grip_range)] * 4 + [90, 120]
        self.lower_ranges = [5, 5, 5, 5, 5, 5]

        ## Data arrays for position
        ## Safe "last position" for hand to start at
        self.pos = [15] * 6
        self.pos[5] = -self.pos[5]
        self.prev_pos = self.pos.copy()

        self.last_pos_msg = [0.0] * 6

        # Data Arrays for touch
        self.touch = [0] * 30
        self.prev_touch = self.touch.copy()

        self.last_cmd_thread = None
        self.last_cmd_event = None

    ## Search for Serial Port to use
    def _setup_serial(self, baud):
        print("Searching for serial ports...")
        com_ports_list = list(list_ports.comports())
        port = ""

        if self.port_idx == -1:
            for i, p in enumerate(com_ports_list):
                if p:
                    if platform.system() == "Linux" or platform.system() == "Darwin":
                        if "CP2102" in str(p):
                            port = p
                            print(f"[{i}] Found: {port}")
                            break
                    elif platform.system() == "Windows":
                        if "COM" in p[0]:
                            port = p
                            print("Found:", port)
                            break
            if not port:
                print("No port found")
                quit()
        else:
            print("Found:", str(com_ports_list[self.port_idx]))
            port = com_ports_list[self.port_idx]

        try:
            print("Connecting...")
            self.ser = serial.Serial(port[0], baud, timeout=0.02)
            print("Connected!")
        except:
            print("Failed to Connect!")
            assert False

        ## Upsample the thumb rotator
        msg = create_misc_msg(0xC2)
        self.ser.write(msg)

        ##Clear Buffer to start
        self.ser.reset_input_buffer()

    def _serial_comm(self, pos_cmd, debug=False):
        """
        Send control command and read latest data.
        pos_cmd: an array of length 6 in range [0, 1]
        """
        ## Get message with new positions
        if pos_cmd is None:
            pos_cmd = np.random.uniform(0, 1, size=6)
        pos_msg = self._process_pos_cmd(pos_cmd)
        self.last_pos_msg = pos_msg

        if debug:
            print("Last pos:", self.pos)
            print("Pos cmd:", pos_cmd)
            print("Pos msg:", pos_msg)

        ## Send Message
        msg = self._generate_tx(pos_msg)
        self.ser.write(msg)

        ## Upsample the thumb rotator
        msg = create_misc_msg(0xC2)
        self.ser.write(msg)

        ## Read first response byte - format header
        data = self.ser.read(1)
        if len(data) == 1:
            reply_format = data[0]
            if (reply_format & 0xF) == 2:
                # reply variant 3
                reply_len = 38
            else:
                # reply variant 1 or 2
                reply_len = 71
            ## Read the rest of the data
            data = self.ser.read(reply_len)
            need_reset = False
            if len(data) == reply_len:
                ## Verify Checksum
                sum = reply_format
                for byte in data:
                    sum = (sum + byte) % 256

                if sum != 0:
                    need_reset = True
                else:
                    ## Extract Position Data
                    ## Position Data is included in all formats in same way
                    ## So we can safely do this no matter the format
                    for i in range(0, 6):
                        raw_data = struct.unpack("<h", data[i * 4 : 2 + (i * 4)])[0]
                        self.pos[i] = raw_data * 150 / 32767

                        ## Bad data, reset serial device - probably framing error
                        if self.pos[i] > 150:
                            need_reset = True

                    ## Extract Touch Data if Available
                    if reply_len == 71:
                        ## Extract Data two at a time
                        for i in range(0, 15):
                            dual_data = data[(i * 3) + 24 : ((i + 1) * 3) + 24]
                            data1 = struct.unpack("<H", dual_data[0:2])[0] & 0x0FFF
                            data2 = (
                                struct.unpack("<H", dual_data[1:3])[0] & 0xFFF0
                            ) >> 4
                            self.touch[i * 2] = int(data1)
                            self.touch[(i * 2) + 1] = int(data2)
                            if data1 > 4096 or data2 > 4096:
                                need_reset = True
            else:
                need_reset = True
        else:
            need_reset = True

        if need_reset:
            self.ser.reset_input_buffer()
            self.reset_count += 1
            need_reset = False
            self.pos = self.prev_pos.copy()
            self.touch = self.prev_touch.copy()

        self.prev_pos = self.pos.copy()
        self.prev_touch = self.touch.copy()

        self.total_count += 1

    def _process_pos_cmd(self, pos_cmd):
        """
        pos_cmd: a list of floats [0, 1] normalized command from external input devices
        """
        assert len(pos_cmd) == 6
        positions = [1] * 6
        for i in range(6):
            positions[i] = self.lower_ranges[i] + pos_cmd[i] * (
                self.upper_ranges[i] - self.lower_ranges[i]
            )
            if i == 5:
                # invert thumb rotator
                positions[i] = -positions[i]
        return positions

    def _pos_to_cmd(self, pos):
        """
        pos: desired hand degrees
        """
        assert len(pos) == 6
        cmd = [0] * 6
        for i in range(6):
            if i == 5:
                pos[i] = -pos[i]
            cmd[i] = (pos[i] - self.lower_ranges[i]) / (
                self.upper_ranges[i] - self.lower_ranges[i]
            )
        return cmd

    def _generate_tx(self, positions):
        """
        Position control mode - reply format must be one of the 3 variants (0x10, 0x11, 0x12)
        """
        txBuf = []

        ## Address in byte 0
        txBuf.append((struct.pack("<B", self.hand_addr))[0])  # device ID

        ## Format Header in byte 1
        txBuf.append((struct.pack("<B", self.reply_mode))[0])

        ## Position data for all 6 fingers, scaled to fixed point representation
        for i in range(0, 6):
            posFixed = int(positions[i] * 32767 / 150)
            txBuf.append((struct.pack("<B", (posFixed & 0xFF)))[0])
            txBuf.append((struct.pack("<B", (posFixed >> 8) & 0xFF))[0])

        ## calculate checksum
        cksum = 0
        for b in txBuf:
            cksum = cksum + b
        cksum = (-cksum) & 0xFF
        txBuf.append((struct.pack("<B", cksum))[0])

        return txBuf

    def close(self):
        self.ser.close()
        print("Completed with " + str(self.reset_count) + " serial device resets")
        print("Total Runs: " + str(self.total_count))

    ### APIs for teleop integration

    def connect(self, hostname=None, port=None):
        self._setup_serial(self.baud_rate)

    def move(self, position, speed=None, force=None, debug=False):
        if self.last_cmd_thread is not None:
            self.last_cmd_event.set()
            self.last_cmd_thread.join()

        self.last_cmd_event = threading.Event()
        self.last_cmd_thread = threading.Thread(
            target=call_gripper, args=(self, position, self.last_cmd_event, debug)
        )

        # print("Ready to start")
        self.last_cmd_thread.start()

        # print("Exit")
        # legacy
        # self._serial_comm(pos_cmd=position, debug=debug)

    def get_current_state(self):
        # last_pos_msg (6,) - unnormalized float in degrees
        # pos (6,) - unnormalized float in degrees
        # touch (30,) - unnormalized int, typically > 1000 means contact
        return self.last_pos_msg + self.pos + self.touch

    def get_current_touch(self):
        return self.touch

    def get_current_position(self):
        # pos readings
        return self.pos
