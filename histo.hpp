#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

MatND getHistogram(const Mat &image, int nbins = 256) {
    int histSize[1] = { nbins };
    float hranges[2] = { 0, (float)nbins };
    const float*ranges[1] = { hranges };
    MatND hist;
    int channels[1] = { 0 };
    //Compute histogram
    calcHist(&image,
        1,//histogram of 1imageonly
        channels,//the channel used
        Mat(),//no mask is used
        hist,//the resulting histogram
        1,//it is a 1D histogram
        histSize,//number of bins
        ranges//pixel value range
    );
    return hist;
}

Mat createHistImage(const MatND &hist, int nbins = 256) {
    double maxVal = 0, minVal = 0;
    minMaxLoc(hist, &minVal, &maxVal, 0, 0);
    //Image on which to display histogram
    Mat histImg(nbins, nbins, CV_8U, Scalar::all(255));
    //set highest point at90%of nbins
    int hpt = static_cast<int>(0.9 * nbins);
    //Draw vertical line for each bin
    //	Scalar color=Scalar(255,0,0,0);//Blue
    for (int h = 0; h < nbins; h++) {
        float binVal = hist.at<float>(h);
        int intensity = static_cast<int>(binVal*hpt / maxVal);
        line(histImg, Point(h, nbins), Point(h, nbins - intensity), Scalar::all(0));
    }
    return histImg;
}

void equalization(Mat& src, Mat& dst, int nbins = 255) {
    src.copyTo(dst);
    int row = dst.rows, col = dst.cols;
    float* val = (float*)malloc(nbins * sizeof(float));
    memset(val, 0, nbins * sizeof(float));

    MatND hist = getHistogram(dst);
    float sum = 0;
    for (int i = 0; i < nbins; i++) {
        sum += hist.at<float>(i);
        *(val + i) = sum;
    }

    uchar getValue = 0;
    uchar* dst_data = dst.data;
    for (int j = 0; j < row * col; j++) {
        getValue = *(dst_data + j);
        *(dst_data + j) = (uchar)round((*(val + getValue) / sum) * nbins);
    }
    free(val);
}

void equalize_st(Mat& image, Mat& dst, int range_min = 0, int range_max = 255) {
    image.copyTo(dst);
    double min = 0, max = 255;
    minMaxLoc(dst, &min, &max);
    int row = dst.rows, col = dst.cols;
    uchar* data = dst.ptr<uchar>(0);
    for (int i = 0; i < row * col; i++) {
        *data++ = (uchar)((((float)(range_max - range_min) / (float)(max - min)) * (float)(*data - min)) + range_min);
    }
}

