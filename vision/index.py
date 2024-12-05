import socket
import detect
import pickle
import struct
import math

def start_tcp_server(host='0.0.0.0', port=2025):
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    point_memory = []
    # Bind the socket to the address and port
    server_socket.bind((host, port))
    
    # Listen for incoming connections
    server_socket.listen(5)  # Allows up to 5 queued connections
    print(f"Server started. Listening on {host}:{port}")
    mode = "NORM"
    while True:
        # Wait for a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")
        
        try:
            while True:
                # Receive data in chunks
                data = client_socket.recv(1024)  # Buffer size
                print(repr(data))
                if not data:
                    print(f"Connection closed by {client_address}")
                    break
                recv = struct.unpack('>I', data)[0]
                print(f"Received: {recv}")
                ## 99991 - 99999
                if recv == 99991: #set Point
                    points = detect.run_webcam_inference_with_centers()
                    point_memory = points
                    client_socket.send(str(f"({len(points)})").encode())
                if recv == 99992:
                    points = detect.run_webcam_inference_with_centers()
                    client_socket.sendall(str(points).encode())
                if recv < 99991:
                    if len(point_memory) == 0:
                        client_socket.sendall(str(f"()").encode())
                        break
                    index = recv
                    print(f"Request points: {index}")
                    print(repr(point_memory))
                    points = point_memory[int(index)]
                    print(f"x: {points[0]}, y: {points[1]}, angle: {points[2]}, radians: {math.radians(points[2])}")
                    client_socket.sendall(str(f"({points[0]}, {points[1]}, {(points[2])})").encode())                    
                # while mode == "REC":
                #     data2 = client_socket.recv(1024)
                #     if not data:
                #         print(f"Connection closed by {client_address}")
                #         break
                #     recv2 = data2.decode('utf-8')
                #     index = int(recv2)
                #     print(f"Request points: {index}")
                #     print(repr(point_memory))
                #     points = point_memory[int(index)]
                #     print(f"x: {points[0]}, y: {points[1]}")
                #     client_socket.sendall(str(f"({points[0]}, {points[1]})").encode())
                    
                    
                # Echo the received data back to the client
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up the connection
            client_socket.close()
        if 0xFF == ord('q'):
            break

if __name__ == "__main__":
    start_tcp_server()
