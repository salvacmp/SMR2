import pyrealsense2 as rs

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Get the depth stream's intrinsics
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()

# Camera matrix components
fx = intrinsics.fx  # Focal length in x direction
fy = intrinsics.fy  # Focal length in y direction
cx = intrinsics.ppx  # Principal point x
cy = intrinsics.ppy  # Principal point y

# Camera matrix
camera_matrix = [
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
]

print("Camera Matrix:")
for row in camera_matrix:
    print(row)

pipeline.stop()
