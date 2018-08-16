#include <opencv2\opencv.hpp>
#include "face_detection.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat in_img0 = imread("./resource/jws_0.PNG", IMREAD_GRAYSCALE);
    Mat in_img1 = imread("./resource/kdw_0.PNG", IMREAD_GRAYSCALE);
    Mat in_img2 = imread("./resource/lsk_0.PNG", IMREAD_GRAYSCALE);
    Mat in_img3 = imread("./resource/qqq.PNG", IMREAD_GRAYSCALE);
    Mat in_img4 = imread("./resource/khr_0.PNG", IMREAD_GRAYSCALE);
    Mat in_img5 = imread("./resource/jws_1.PNG", IMREAD_GRAYSCALE);
    Mat in_img6 = imread("./resource/khr_1.PNG", IMREAD_GRAYSCALE);

    CompareLBP test0(in_img5, "jws_1.PNG", in_img0, "jws_0.PNG");
    CompareLBP test1(in_img5, "jws_1.PNG", in_img1, "kdw_0.PNG");
    CompareLBP test2(in_img5, "jws_1.PNG", in_img2, "lsk_0.PNG");
    CompareLBP test3(in_img5, "jws_1.PNG", in_img3, "qqq.PNG");
    CompareLBP test4(in_img5, "jws_1.PNG", in_img4, "khr_0.PNG");
    ContainerProcess con;
    con.add(&test0);
    con.add(&test1);
    con.add(&test2);
    con.add(&test3);
    con.add(&test4);
    con.run();

    test0.setParam1(in_img6, "khr_1.PNG");
    test1.setParam1(in_img6, "khr_1.PNG");
    test2.setParam1(in_img6, "khr_1.PNG");
    test3.setParam1(in_img6, "khr_1.PNG");
    test4.setParam1(in_img6, "khr_1.PNG");
    ContainerProcess con2;
    con2.add(&test0);
    con2.add(&test1);
    con2.add(&test2);
    con2.add(&test3);
    con2.add(&test4);
    con2.run();
    
    return 0;
}