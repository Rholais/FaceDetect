#include "stdafx.h"
#include "../FaceDetect.h"
#include <windows.h>
#include "precomp.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

#ifdef _EiC
#define WIN32
#endif
#ifdef _DEBUG
#pragma comment(lib, "LibOPCV/lib/opencv_core246d.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_highgui246d.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_imgproc246d.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_objdetect246d.lib")
#else
#pragma comment(lib, "LibOPCV/lib/opencv_core246.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_highgui246.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_imgproc246.lib")
#pragma comment(lib, "LibOPCV/lib/opencv_objdetect246.lib")
#endif

using namespace cv;

static CvMemStorage* storage = NULL;
static CvHaarClassifierCascade* cascade = NULL;
static Mat frame;
static IplImage *frame_copy = NULL;
static CvRect* roi = NULL;

static VideoCapture *cap;

int detect_and_draw( CvArr* _img, int zoom, int dat[]);
void sphere_to_decare(int dat[], float *position, CvSize size);

const char* cascade_name =
"haarcascade_frontalface_alt.xml";
/*    "haarcascade_profileface.xml";*/

static double scale = 1;

double t = 0;

const unsigned int FD_CAMARA = 1;
const unsigned int FD_RENEW = 2;
const unsigned int FD_REGION = 4;
const unsigned int FD_LOCKED = 8;
const unsigned int FD_RETURN = 16;

static double pos[][4] = 
{
	{ 0.0, 0.0, 0.0, 0.0 },
	{ 0.0, 0.0, 0.0, 0.0 },
	{ 0.0, 0.0, 0.0, 0.0 },
};

void OpenCVInit()
{
	cascade_name = "haarcascade_frontalface_alt2.xml";
	//opencv装好后haarcascade_frontalface_alt2.xml的路径,
	//也可以把这个文件拷到你的工程文件夹下然后不用写路径名cascade_name= "haarcascade_frontalface_alt2.xml";  
	//或者cascade_name ="C:\\Program Files\\OpenCV\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"

	cascade = static_cast<CvHaarClassifierCascade*>(cvLoad( cascade_name, NULL, NULL, NULL ));

	if( !cascade )
	{
		MessageBox(0,"ERROR: Could not load classifier cascade\n","",MB_OK);
		fprintf_s( stderr, "ERROR: Could not load classifier cascade\n" );
		fprintf_s( stderr,
			"Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
		return ;
	}
	storage = cvCreateMemStorage();
	if( !storage )
	{
		MessageBox(0,"ERROR: Could not create memory storage\n","",MB_OK);
		fprintf_s( stderr, "ERROR: Could not create memory storage\n" );
		return ;
	}
}
void ReleaseOpenCV()
{
	if( cascade )
	{
		cvReleaseHaarClassifierCascade(&cascade);
	}
	if ( storage )
	{
		cvReleaseMemStorage( &storage );
	}
}
//
FACEDETECT_API int FaceDetectOP(int width, int height, int zoom, void* buf, int dat[])
//int main( int argc, char** argv )
{
	IplImage *frame_copy = NULL;

	//scale = zoom;

	frame_copy = cvCreateImage( cvSize(width/zoom,height/zoom),IPL_DEPTH_8U, 3 );
	//memcpy(frame_copy->imageData, (char*)buf, frame_copy->imageSize);

	//int* data = dat;
	unsigned char* pBuffer = (unsigned char*)buf;
	unsigned char* ptr = (unsigned char*)frame_copy->imageData;
	int nLineBytes = (width * 24 + 31) / 32 * 4;
	int k = 0;
	for(int j = 0; j < height; j +=1*zoom)
	{
		unsigned char* g = ptr;
		unsigned char* rgb = pBuffer + (height - j - 1) * nLineBytes;
		for(int i = 0; i < width; i += 1*zoom, rgb += 3*zoom, g+=3)
		{

			g[0] = rgb[0];
			g[1] = rgb[1];
			g[2] = rgb[2];
		}
		ptr += frame_copy->widthStep;//
	}
	int fn = detect_and_draw( frame_copy , zoom, dat);


	cvReleaseImage( &frame_copy );

	return fn;
}

int detect_and_draw( CvArr *_img , int zoom, int dat[])
{
	IplImage *img = (IplImage*)_img;
	int fn= 0;
	IplImage *gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
	cvCvtColor( img, gray, CV_BGR2GRAY );
	IplImage *small_img;
	if(roi)
	{
		cvSetImageROI(gray, *roi);

		if(roi->height * roi->width >= 225000)
		{
			zoom = 4;
		}
		else if(roi->height * roi->width >= 150000)
		{
			zoom = 3;
		}
		else if(roi->height * roi->width >= 75000)
		{
			zoom = 2;
		}
		else
		{
			zoom = 1;
		}

		small_img = cvCreateImage( cvSize( D2I(roi->width/zoom),
		D2I(roi->height/zoom)),
		8, 1 );
	}
	else
	{
		small_img = cvCreateImage( cvSize( D2I(img->width/zoom),
		D2I(img->height/zoom)),
		8, 1 );
	}
	cvResize( gray, small_img, CV_INTER_LINEAR );
	cvEqualizeHist( small_img, small_img );
	
	cvClearMemStorage( storage );
	//MessageBox(0,"Clear Succeed.","",MB_OK);
	if( cascade )
	{
		CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,
			1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT/*CV_HAAR_DO_CANNY_PRUNING*/,
			cvSize(30, 30) );
		int k=0;
		if(faces && faces->total >= 1)
		{
			for(int i = 0; i != faces->total; i++ )
			{
				CvRect* r = reinterpret_cast<CvRect*>(cvGetSeqElem( faces, i ));
				dat[k++] = r->x*zoom;
				dat[k++] = r->y*zoom;
				dat[k++] = r->width*zoom;
				dat[k++] = r->height*zoom;
				if(roi)
				{
					dat[k-4] += roi->x;
					dat[k-3] += roi->y;
				}
				fn++;

				roi = new CvRect(cvRect(dat[k-4] - dat[k-2] / 2, dat[k-3] - dat[k-1] / 2, dat[k-2] * 2, dat[k-1] * 2));
				if(roi->x < 0)
				{
					roi->width -= -(roi->x);
					roi->x = 0;
				}
				if(roi->y < 0)
				{
					roi->width -= -(roi->y);
					roi->y = 0;
				}
				if(roi->x + roi->width > img->width)
				{
					roi->width -= img->width - roi->x;
				}
				if(roi->y + roi->height > img->height)
				{
					roi->height-= img->height - roi->y;
				}

			}
		}
		else
		{
			roi = new CvRect(cvRect( 0, 0, img->width, img->height ));
		}

	}

	//cvShowImage( "result", img );
	cvReleaseImage( &gray );
	cvReleaseImage( &small_img );
	return fn;
}

FACEDETECT_API int FdWidth = 0;
FACEDETECT_API int FdHeight = 0;

FACEDETECT_API void fdGetPos(float* position, unsigned int flag = 0)
{
	int* dat = new int[4];
	for(int i = 0; i != 4; ++i)
	{
		dat[i] = 0;
	}
	try
	{
		cap = new VideoCapture(0);
		if(!cap)
		{
			throw(std::exception("Capture Failed."));
		}
	}
	catch(std::exception ex)
	{
		MessageBox(0,ex.what(),"",MB_OK);
	}
	if( cap )
	{
		while(! (flag & FD_RETURN))
		{
			if( flag & FD_CAMARA )
			{
				if( !cap->grab())
				{
					break;
				}
				cap->retrieve(frame);
				if(!FdWidth || !FdHeight)
				{
					FdWidth = frame.cols;
					FdHeight = frame.rows;
				}
				if( !frame_copy )
				{
					frame_copy = cvCreateImage( cvSize(FdWidth, FdHeight),
						IPL_DEPTH_8U, 3 );
				}
				CvMat arr = frame;
				frame_copy = cvGetImage(&arr, frame_copy);
				//if( frame->origin == IPL_ORIGIN_TL )
				//{
				//		cvCopy( frame, frame_copy, 0 );
				//}
				//else
				//{
				//		cvFlip( frame, frame_copy, 0 );
				//}
				//MessageBox(0,"Retrieve Succeed.","",MB_OK);

				detect_and_draw( frame_copy , D2I(scale), dat);
				if(flag | FD_REGION)
				{
					flag |= FD_LOCKED;
					for(int i = 0; i != 4; ++i)
					{
						pos[2][i] = dat[i];
					}
					flag &=  (~FD_LOCKED);
				}
				else
				{
					sphere_to_decare(dat, position, cvSize(FdWidth, FdHeight));
				}
				for(int i = 0; i != 4; ++i)
				{
					position[i] = 0.1 * pos[0][i] + 0.3 * pos[1][i] + 0.6 * pos[2][i];
					pos[0][i] = pos[1][i];
					pos[1][i] = pos[2][i];
				}

				flag |= FD_RENEW;
			}
		}
	}
	delete [] dat;
}

// 将摄像头截图的相对坐标转换为三维直角坐标
void sphere_to_decare(int dat[], float *position, CvSize size)
{
	pos[2][2] = 284.8 / (dat[2] + dat[3]);
	pos[2][0] = (dat[0] + dat[2] / 2 - size.width / 2) / 36.3 / position[2];
	pos[2][1] = (dat[1] + dat[3] / 2 - size.height / 2) / 36.3 / position[2];
}