// FaceDetect.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "FaceDetect.h"
#include "LibCCV/ccv.h"

// This is an example of an exported variable
FACEDETECT_API int nFaceDetect=0;
extern ccv_bbf_classifier_cascade_t* cascade;

// This is an example of an exported function.
FACEDETECT_API int FaceDetect(int width, int height, unsigned char* buffer, int* data)
{
	ccv_dense_matrix_t* image = 0;
	int ctype = CCV_8U | CCV_C1;
	image = ccv_dense_matrix_new(height, width, ctype | CCV_NO_DATA_ALLOC, buffer, 0);
	ccv_array_t* faces = ccv_bbf_detect_objects(image, &cascade, 1, ccv_bbf_default_params);
	int i,k = 0;
	for (i = 0; i < faces->rnum; i++)
	{
		ccv_comp_t* face = (ccv_comp_t*)ccv_array_get(faces, i);
		data[k++] = face->rect.x;
		data[k++] = face->rect.y;
		data[k++] = face->rect.width;
		data[k++] = face->rect.height;
	}
	k = faces->rnum;
	ccfree(image);
	ccv_array_free(faces);
	return k;
}

unsigned char buffer[768000];

FACEDETECT_API int FaceDetectEx(int width, int height, int zoom, void* buf, int dat[])
{
	int ZOOM_OUT = zoom;
	int fn;
    {
        int* data = dat;
        unsigned char* pBuffer = (unsigned char*)buf;
        unsigned char* ptr = buffer;
        int nLineBytes = (width * 24 + 31) / 32 * 4;
        int k = 0;
        for(int j = 0; j < height; j +=1*ZOOM_OUT)
        {
            unsigned char* g = ptr;
            unsigned char* rgb = pBuffer + (height - j - 1) * nLineBytes;
            for(int i = 0; i < width; i += 1*ZOOM_OUT, rgb += 3*ZOOM_OUT, g++)
            {
				
	            *g = (unsigned char)((rgb[0] * 6969 + rgb[1] * 23434 + rgb[2] * 2365) >> 15);
	            if ( *g > 180 )
		            *g = ((*g)>>2)+90;
	            else if ( *g > 90 )
		            *g = ((*g)>>1)+45;
            }
            ptr += width/ZOOM_OUT;
        }
        fn = FaceDetect(width/ZOOM_OUT, height/ZOOM_OUT, buffer, data);
        for (int i = 0; i < fn*4; i++ )
        {
            data[i] *=ZOOM_OUT;
        }
        //m_pVideoAnaView->PostMessage(WM_FACE_DETECT, (WPARAM)data, fn);
        //TRACE1("fn = %d\n",fn);
    }
	return fn;
}
// This is the constructor of a class that has been exported.
// see FaceDetect.h for the class definition
CFaceDetect::CFaceDetect()
{
	return;
}
