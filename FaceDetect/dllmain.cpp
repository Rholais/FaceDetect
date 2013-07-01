// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "LibCCV/ccv.h"
void OpenCVInit();
void ReleaseOpenCV();

ccv_bbf_classifier_cascade_t* cascade;
BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		cascade = ccv_load_bbf_classifier_cascade("D:/face");
		OpenCVInit();
		break;
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		ccv_bbf_classifier_cascade_free(cascade);
		ReleaseOpenCV();
		break;
	}
	return TRUE;
}

