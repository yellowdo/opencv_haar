#pragma once
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define PLUS 0
#define MINUS 1
#define MULTI 2
#define DIVIDE 3

void process_pointer(Mat& image, Mat& dst, int op, int val = 64) {
    image.copyTo(dst);
    int row = dst.rows, col = dst.cols, channel = dst.channels();
    int calc = 0;
    uchar* data = dst.data;
    for (int i = 0; i < row * col * channel; i++) {
        calc = (int)*(data + i);
        switch (op) {
            case PLUS: calc += val; break;
            case MINUS: calc -= val; break;
            case MULTI: calc *= val; break;
            case DIVIDE: calc /= val; break;
            default: break;
        }
        if (calc > 255) calc = 255;
        else if (calc < 0) calc = 0;
        *(data + i) = (uchar) calc;
    }
}