import socket
import time
import struct

HOST = "192.168.0.43"
PORT = 30002


def sendMove(x, y):
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    
    s.send((f"movel(p[{x}, {y}, 0.2, 0.572354, -3.07864, -0.0776772], a=1, v=1, r=0)" + "\n").encode('utf8'))
    
    data = s.recv(1024)
    decoded = struct.unpack('!6f', data[:24])
    print(repr(decoded))
    s.close()
    
    