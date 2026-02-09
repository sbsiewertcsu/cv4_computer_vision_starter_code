/*
 *
 *  Example by Sam Siewert 
 *
 *  Updated 02/08/26 for OpenCV 4.11
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

//#define HRES 640
//#define VRES 480


int main( int argc, char** argv )
{
    //cvNamedWindow("Capture Example", CV_WINDOW_AUTOSIZE);
    //CvCapture* capture = cvCreateCameraCapture(0);
    Mat mat_frame, mat_hsv;
    //IplImage* frame;
    VideoCapture vcap;

    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);

    //open the video stream and make sure it's opened
    // "0" is the default video device which is normally the built-in webcam
    if(!vcap.open(0)) 
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    else
    {
	   std::cout << "Opened default camera interface" << std::endl;
    }


    while(1)
    {
        //frame=cvQueryFrame(capture);
	if(!vcap.read(mat_frame)) {
		std::cout << "No frame" << std::endl;
		cv::waitKey();
	}
        //if(!frame) break;
	
	cv::cvtColor(mat_frame, mat_hsv, COLOR_BGR2HSV);

	cv::imshow("RGB Color Example", mat_frame);
	cv::imshow("HSV Color Example", mat_hsv);

        char c = waitKey(10);
        if( c == 'q' ) break;
    }

    //cvReleaseCapture(&capture);
    //cvDestroyWindow("Capture Example");
    
};
