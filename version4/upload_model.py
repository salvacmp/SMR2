import roboflow

rf = roboflow.Roboflow(api_key="1puTfiKDYQCiVOwkPYOH")
project = rf.workspace().project("brick-a")

#can specify weights_filename, default is "weights/best.pt"
version = project.version("1")
version.deploy("yolov11", "./", "best.pt")