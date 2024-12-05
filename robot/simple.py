import socket
import time
import struct
from modBusClient import URModbusServer
HOST = "192.168.0.43"  # Replace with your robot's IP address
PORT = 30002  # Port for secondary client interface

# while True:
    # Establish the connection
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((HOST, PORT))

    # # Send the command to get TCP force
    # s.send(("get_tcp_force()" + "\n").encode('utf8'))
    # time.sleep(1)

    # # Receive the data
    # data = s.recv(2048)  # Adjust buffer size if necessary
    # s.close()

    # # Print raw data
    # # print("Raw Data:", repr(data))

    # # Decode the data
    # # Assuming the response contains six floats representing Fx, Fy, Fz, Tx, Ty, Tz
    # try:
    #     # Each float is 4 bytes, so we expect 24 bytes for 6 floats
    #     decoded_data = struct.unpack('!6f', data[:24])  # Big-endian float format
    #     print( decoded_data)
    # except struct.error as e:
    #     print("Error decoding data:", e)
UR = URModbusServer(HOST)
while True:
    print(UR.get_tcp_position())
