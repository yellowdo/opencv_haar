#include <opencv2/opencv.hpp>
#include "face_detection.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("./resource/face_ex.jpg", IMREAD_GRAYSCALE);
    resize(image, image, Size(220, 300));
    imshow("Original image", image);

    Mat intg_image;
    integralImage(image, intg_image);
    cout << haarLikeValue(intg_image, 76, 182, 24, 24) << endl;

    Mat cube(5, 5, CV_8U);
    for (int i = 0; i < cube.rows * cube.cols; i++) {
        cube.at<uchar>(i) = i + 1;
    }
    cout << cube << endl;
    Mat intg;
    integralImage(cube, intg);
    cout << intg << endl;
    cout << haarLikeValue(intg, 1, 1, 4, 4) << endl;

    waitKey();
    return 0;
}