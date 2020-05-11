# Camera Pose Estimation using least squares method

Given instrinsic parameters and 3D points in a (world) coordinate frame (defined at the corner of the box),
find the pose of the camera in that coordinate frame.

The solution described here is using Gauss-Newton method to iteratively find the roll, pitch, yaw, x, y, z of 
the world frame from camera frame. 

## Execution steps

```
mkdir build
cd build
cmake ..
make
./main
```


![alt text](images/image.jpg)
### Pose estimation after step 1
![alt text](images/0.jpg)
### Pose estimation after step 2
![alt text](images/1.jpg)
### Pose estimation after step 3
![alt text](images/2.jpg)
### Pose estimation after step 4
![alt text](images/3.jpg)
### Pose estimation after step 5
![alt text](images/4.jpg)
