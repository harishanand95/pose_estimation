cmake_minimum_required(VERSION 2.8)     
PROJECT(pose_estimation_1)

FIND_PACKAGE( OpenCV REQUIRED )           
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

INCLUDE_DIRECTORIES( /usr/include/eigen3)
ADD_EXECUTABLE(main src/main.cpp)                
TARGET_LINK_LIBRARIES (main ${OpenCV_LIBS}) 
