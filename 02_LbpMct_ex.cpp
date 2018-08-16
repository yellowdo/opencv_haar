#include <opencv2\opencv.hpp>
#include "scala_calc.hpp"
#include "histo.hpp"
#include "face_detection.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("./resource/face_ex.jpg", IMREAD_GRAYSCALE);
    resize(image, image, Size(220, 300));
    imshow("Original image", image);

    Mat lbp_image;
    parallel_for_(Range(1, image.rows - 1), Parallel_LBP_MAT(image, lbp_image));
    imshow("LBP image", lbp_image);

    Mat range, minus80, plus80;
    Mat lbp_range, lbp_minus80, lbp_plus80;
    equalize_st(image, range, 80, 175);
    process_pointer(range, minus80, MINUS, 80);
    process_pointer(range, plus80, PLUS, 80);
    imshow("Range 80~175 image", range);
    imshow("-80 image", minus80);
    imshow("+80 image", plus80);

    parallel_for_(Range(1, range.rows - 1), Parallel_LBP_MAT(range, lbp_range));
    parallel_for_(Range(1, minus80.rows - 1), Parallel_LBP_MAT(minus80, lbp_minus80));
    parallel_for_(Range(1, plus80.rows - 1), Parallel_LBP_MAT(plus80, lbp_plus80));
    imshow("LBP Range 80~175 image", lbp_range);
    imshow("LBP -80 image", lbp_minus80);
    imshow("LBP +80 image", lbp_plus80);

    Mat mct_image;
    parallel_for_(Range(1, image.rows - 1), Parallel_MCT_MAT(image, mct_image));
    imshow("MCT image", mct_image);

    waitKey();
    return 0;
}