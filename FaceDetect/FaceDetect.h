// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the FACEDETECT_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// FACEDETECT_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef FACEDETECT_EXPORTS
#define FACEDETECT_API __declspec(dllexport)
#else
#define FACEDETECT_API __declspec(dllimport)
#endif

// This class is exported from the FaceDetect.dll
class FACEDETECT_API CFaceDetect {
public:
	CFaceDetect(void);
	// TODO: add your methods here.
};

extern FACEDETECT_API int nFaceDetect;

FACEDETECT_API int FaceDetect(int width, int height, unsigned char* buffer, int* data);
FACEDETECT_API int FaceDetectEx(int width, int height, int zoom, void* buf, int dat[]);
FACEDETECT_API int FaceDetectOP(int width, int height, int zoom, void* buf, int dat[]);
FACEDETECT_API void fdGetPos(float* position, unsigned int flag);
