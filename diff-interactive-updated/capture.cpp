/*
 *
 *  Example by Sam Siewert 
 *
 *  Updated 12/6/18 for OpenCV 3.1
 *
 *  Updated 8/1/23 for OpenCV 4.x
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <syslog.h>
#include <time.h>

//#include "opencv2/opencv.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

char difftext[20];
char timetext[20];

int main( int argc, char** argv )
{
    Mat mat_frame, mat_gray, mat_diff, mat_gray_prev;
    VideoCapture vcap;
    unsigned int diffsum, maxdiff, framecnt=0;
    double percent_diff=0.0, percent_diff_old = 0.0;
    double ma_percent_diff = 0.0, fcurtime=0.0, start_fcurtime=0.0;

    struct timespec curtime;


    clock_gettime(CLOCK_REALTIME, &curtime);
    start_fcurtime = (double)curtime.tv_sec + ((double)curtime.tv_nsec/1000000000.0);

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

    while(!vcap.read(mat_frame)) {
	std::cout << "No frame" << std::endl;
	cv::waitKey(33);
    }
	
    cv::cvtColor(mat_frame, mat_gray, COLOR_BGR2GRAY);

    mat_diff = mat_gray.clone();
    mat_gray_prev = mat_gray.clone();

    maxdiff = (mat_diff.cols)*(mat_diff.rows)*255;

    while(1)
    {
	if(!vcap.read(mat_frame)) {
		std::cout << "No frame" << std::endl;
		cv::waitKey();
	}
        else
        {
            framecnt++;
            clock_gettime(CLOCK_REALTIME, &curtime);
            fcurtime = (double)curtime.tv_sec + ((double)curtime.tv_nsec/1000000000.0) - start_fcurtime;
        }
	
	cv::cvtColor(mat_frame, mat_gray, COLOR_BGR2GRAY);

	absdiff(mat_gray_prev, mat_gray, mat_diff);

	// worst case sum is resolution * 255
	diffsum = (unsigned int)cv::sum(mat_diff)[0]; // single channel sum

	percent_diff = ((double)diffsum / (double)maxdiff)*100.0;

        if(framecnt < 3)
            ma_percent_diff=(percent_diff+percent_diff_old)/(double)framecnt;
        else
            ma_percent_diff = ( (ma_percent_diff * (double)framecnt) + percent_diff ) / (double)(framecnt+1);

        //printf("percent diff=%lf, old=%lf, ma=%lf, cnt=%u, change=%lf\n", percent_diff, percent_diff_old, ma_percent_diff, framecnt, (percent_diff - percent_diff_old));


        syslog(LOG_CRIT, "TICK: percent diff, %lf, old, %lf, ma, %lf, cnt, %u, change, %lf\n", percent_diff, percent_diff_old, ma_percent_diff, framecnt, (percent_diff - percent_diff_old));
        sprintf(difftext, "%8d",  diffsum);
        sprintf(timetext, "%6.3lf",  fcurtime);

        percent_diff_old = percent_diff;

        // tested in ERAU Jetson lab
	if(percent_diff > 0.5)
        {
            cv::putText(mat_diff, difftext, Point(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200,200,250), 1, LINE_AA);
            cv::putText(mat_diff, timetext, Point(500,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200,200,250), 1, LINE_AA);
        }

        //if(percent_diff > 0.5) printf("TICK @ %lf\n", fcurtime);

	cv::imshow("Clock Current", mat_gray);
	cv::imshow("Clock Previous", mat_gray_prev);
	cv::imshow("Clock Diff", mat_diff);


        char c = cv::waitKey(100); // sample rate
        if( c == 'q' ) break;

	mat_gray_prev = mat_gray.clone();
    }

};
