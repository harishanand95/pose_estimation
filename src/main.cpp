/* Harish
 * 2020-05-09
 * Pose estimation using least squares method
 * Based on course work in http://cs-courses.mines.edu/csci507
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <Eigen/Dense>


using namespace cv;
using namespace std;
using namespace Eigen;


void project3Dto2D(const Matrix<double, 4, 6> points3D, const Matrix3d intrinsicMatrix,
                   const Matrix<double, 3, 4> extrinsicMatrix, Matrix<double, 3, 6>& points2D){

    /* points3D is 4x6 (6 points where in x,y,z,1 where 1 is adding translation vector)
     * extrinsicMatrix is 3x4
     * intrinsicMatrix is 3x3
     * Output should be 6 2D points ie, 3x6 matrix.
     */
    points2D = intrinsicMatrix * extrinsicMatrix * points3D;
    int last_row = points2D.rows() - 1;
    points2D = points2D.array().rowwise() / points2D.row(last_row).array();
}

void createPose(const Matrix <double , 6, 1>& pose, Matrix<double, 3, 4>& transformMatrix){

    Eigen::AngleAxisd rollAngle(pose(0, 0), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pose(1, 0), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(pose(2, 0), Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> q = yawAngle * pitchAngle * rollAngle;

    Eigen::Matrix3d rotationMatrix = q.matrix();
    Eigen::Matrix4d trans;
    trans.setIdentity();
    trans.block<3, 3>(0,0) = rotationMatrix;
    trans(0, 3) = pose(3, 0);
    trans(1, 3) = pose(4, 0);
    trans(2, 3) = pose(5, 0);
    transformMatrix = trans.block<3, 4>(0,0);

}

int main(){
	
	Mat image;	
	image = imread("../images/image.tif", CV_LOAD_IMAGE_COLOR);
	if(! image.data )
    {
	    cout <<  "Could not open or find the image" << std::endl ;
       	return -1;
    }
    //	imshow("Display window", image);
    //	waitKey(0);


    // Intrinsic matrix (K) definition
    double focalLength = 715;
    double centerX = 354;
    double centerY = 245;
    Matrix3d K;
	K << focalLength, 0, centerX,
	     0, focalLength, centerY,
	     0, 0, 1;


	// The 2D image pixel location of the 6 corners
    Matrix<double, 6, 2> actual2DPoints;
    actual2DPoints << 183, 147,
                      350, 133,
                      454, 144,
                      176, 258,
                      339, 275,
                      444, 286;

    Matrix<double, 4, 6> actual3DPoints;
    actual3DPoints << 0,  0, 2,  0, 0, 2,
                     10,  2, 0, 10, 2, 0,
                      6,  6, 6,  2, 2, 2,
                      1,  1, 1,  1, 1, 1;

    // Define pose and give a guessed pose value
    // Roll, pitch, yaw, x, y, z

    Matrix <double , 6, 1> pose;
    pose << 1.5, -1.0, 0.0, 0, 0, 30;

    // Create extrinsic matrix from this pose
    Matrix<double, 3, 4> RT;
    createPose(pose, RT);


    // Predict the pixel location of the 3D points
    Matrix<double, 3, 6> estimated2DPoints = Matrix<double, 3, 6>::Zero();
    project3Dto2D(actual3DPoints, K, RT, estimated2DPoints);


    // Plot the points
    vector<cv::Point> points;
    for (int i = 0; i < 6; i++ ) {
        int x = (int) estimated2DPoints(0, i);
        int y = (int) estimated2DPoints(1, i);
        points.emplace_back(cv::Point(x, y));
        cv::circle(image, cv::Point(x, y), 4, cv::Scalar(255,0,255), -1);
    }

    // Show the points
    imshow("Display window", image);
    waitKey(0);
    
    return 1;
}
