/**
*  @author Kaushik Roy
*  @brief Reliable Edge map generation feom training image set (<256) using accumulation
*/
#include <windows.h>
#include <vector>
#include "KScOpenCvUtils.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "BasicBgModel.h"

BOOL WINAPI DllMain(HINSTANCE hDLL, DWORD dwReason, LPVOID lpReserved)
{
switch (dwReason)
{
case DLL_PROCESS_ATTACH: break;
case DLL_THREAD_ATTACH:  break;
case DLL_THREAD_DETACH:  break;
case DLL_PROCESS_DETACH: break;
}

return TRUE;
}

cv::Mat mPixelBasedAccumulator;

int findGradBin(const cv::Mat &hist, int histSize, int total, double topPercent)
{
double gradMagTh = 1.0 - topPercent;
float sum = 0.f;
int bin = 0;

// find bin.
for (int i = 0; i < histSize; i++)
{
sum += hist.at<float>(i);

if (sum / total >= gradMagTh)
{
bin = i;
break;
}

}

return bin;
}


int CDECL BasicBgModel(int* frameIndex, int* lastFrameIndex, KScScalarImage2dUint8* srcImg, KScScalarImage2dUint8* dstImg, KScHistogram1d* hist)
{
if (srcImg->GetId() != KS_SCALAR_IMAGE_2D_UINT8)
{
::MessageBox(NULL, "Output buffer type not suitable.",
"iPrewittOutGradientMagPhaseInt8", MB_OK);
return FALSE;
}

if (dstImg->GetId() != KS_SCALAR_IMAGE_2D_UINT8)
{
::MessageBox(NULL, "Output buffer type not suitable.",
"iPrewittOutGradientMagPhaseInt8", MB_OK);
return FALSE;
}

int dx = srcImg->GetMainXSize();
int dy = srcImg->GetMainYSize();

if (!dx || !dy)
{
::MessageBox(NULL, "Input buffer not allocated.",
"iHighPassButterworthFilter", MB_OK);
return FALSE;
}

if (dx != dstImg->GetXSize() || dy != dstImg->GetYSize())
{
dstImg->Free();
if (dstImg->Alloc(dx, dy))
{
::MessageBox(NULL, "Fail to allocate output buffer.",
"iPrewittOutGradientMagPhaseInt16", MB_OK);
return FALSE;
}
}
/*else
{*/
dstImg->InitTo(0);
//}

int scale = 1;
int delta = 0;
int ddepth = CV_16S;

cv::Mat src, dst, blur_img, canny_edge_img;
src = KScScalarImage2dUint8ToMat(srcImg);

//cv::blur( src, blur_img, cv::Size(3,3) );
cv::GaussianBlur(src, blur_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

//Sobel Edge Detection

cv::Mat grad, grad_x, grad_y;
cv::Mat abs_grad_x, abs_grad_y;

/// Gradient X
cv::Sobel(blur_img, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
cv::convertScaleAbs(grad_x, abs_grad_x);

/// Gradient Y
cv::Sobel(blur_img, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
cv::convertScaleAbs(grad_y, abs_grad_y);

/// Total Gradient (approximate)
cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

// Threshold of20% distribution of Edge 

bool uniform = true, accumulate = false;
int histSize = 256;    // bin size
float range[] = { 0, 255 };
const float *ranges[] = { range };

// Calculate histogram
cv::Mat cv_hist;
cv::calcHist(&grad, 1, 0, cv::Mat(), cv_hist, 1, &histSize, ranges, uniform, accumulate);

// Determine threshold to cut off 20% accumulation from gradient distribution
int bin = findGradBin(cv_hist, 256, grad.rows*grad.cols, .20);

// Apply Threshold
dst = KScScalarImage2dUint8ToMat(dstImg);

cv::Mat tmpDst;
cv::threshold(grad, tmpDst, bin, 1, cv::THRESH_BINARY);

if (*frameIndex == 0)
{
	tmpDst.copyTo(mPixelBasedAccumulator);
}
else
{
	cv::addWeighted(tmpDst, 1, mPixelBasedAccumulator, 1, 0, mPixelBasedAccumulator);
}

// END of Sobel Edge Detection

if (*frameIndex == *lastFrameIndex - 1)
{

	double thr = (*lastFrameIndex)*.80;
	cv::threshold(mPixelBasedAccumulator, mPixelBasedAccumulator, thr, 255, cv::THRESH_BINARY);

	mPixelBasedAccumulator.copyTo(dst);

}

MatToKScScalarImage2dUint8(dst, dstImg);
return TRUE;
}
