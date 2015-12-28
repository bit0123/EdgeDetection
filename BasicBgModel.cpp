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

std::vector< std::vector<int> > mBgModelImg;


int CDECL BasicBgModel(int* frameIndex, KScScalarImage2dUint8* srcImg, KScScalarImage2dUint8* dstImg, KScHistogram1d* hist)
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

	cv::Mat src, dst, blur_img;
	src = KScScalarImage2dUint8ToMat(srcImg);

	if (*frameIndex == 0)
	{
		int col = src.cols, row = src.rows;

		mBgModelImg.resize(row, std::vector<int>(col, 0));
	}

	//cv::blur( src, blur_img, cv::Size(3,3) );
	cv::GaussianBlur(src, blur_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	/// Canny detector
	//cv::Canny(blur_img, blur_img, 25, 50); 

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

	/**
	//Manually Histobram Calculation
	cv::Mat cv_hist = cv::Mat::zeros(256, 1, CV_32F);
	//cv::calcHist(&grad, 1, 0, cv::Mat(), cv_hist, 1, &histSize, ranges, uniform, accumulate);

	for (int i = mIgnr; i<grad.rows- mIgnr; i++) {
	for (int j = mIgnr; j<grad.cols- mIgnr; j++) {
	int v = grad.at<int>(i, j);
	if (v > 255)
	v = 255;
	cv_hist.at<float>(v, 0)++;
	}
	}
	*/

	// Determine threshold to cut off 20% accumulation from gradient distribution
	double total, thr = .2, accTotal = 0, bin;
	total = grad.rows * grad.cols;
	for (int h = 0; h < histSize; h++)
	{
		accTotal += cv_hist.at<float>(h);
		if (accTotal / total >= thr)
		{
			bin = h;
			break;
		}
	}

	// Apply Threshold
	dst = KScScalarImage2dUint8ToMat(dstImg);
	cv::threshold(grad, dst, bin, 255, cv::THRESH_BINARY);

	/**
	// Manually applying threshold
	cv::Mat mEdgeImg = cv::Mat::zeros(grad.rows, grad.cols, CV_8UC1);
	//cv::threshold(grad, mEdgeImg, bin, 255, cv::THRESH_BINARY);
	int mIgnr = 1;
	for (int i = mIgnr; i<grad.rows - mIgnr; i++) {
	for (int j = mIgnr; j<grad.cols - mIgnr; j++) {
	int v = grad.at<int>(i, j);
	if (v > bin)
	mEdgeImg.at<unsigned char>(i, j) = 255;
	}
	}
	*/

	// END of Sobel Edge Detection

	MatToKScScalarImage2dUint8(dst, dstImg);

	return TRUE;
}
