#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

#define CAM_MODE 0

void detectAndDisplay(Mat &frame, Mat &frame_gray, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade) {
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	vector<Rect> faces, eyes;
	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.2, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (Rect face_r : faces) {
		rectangle(frame, face_r, Scalar(0, 255, 0), 2, 8, 0);
		Mat faceROI = frame_gray(face_r);

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.2, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (Rect eye_r : eyes) {
			Point center(face_r.x + eye_r.x + eye_r.width * 0.5,
				face_r.y + eye_r.y + eye_r.height * 0.5);
			int radius = cvRound((eye_r.width + eye_r.height) * 0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 2, 8, 0);
		}
	}
	imshow("Capture - Face detection", frame);
}

int main() {
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
    if (!face_cascade.load("../model/haarcascade_frontalface_alt.xml") ||
        !eyes_cascade.load("../model/haarcascade_eye_tree_eyeglasses.xml")) return -1;

	Mat curr_frame, frame_gray;

#if CAM_MODE
	VideoCapture cap(0);
    while (cap.isOpened()) {
        if (!cap.read(curr_frame)) break;        
        detectAndDisplay(curr_frame, frame_gray, face_cascade, eyes_cascade);
        if (waitKey(33) == 27) break; //Halt ESC
    }
#else
	curr_frame = imread("./resource/face_ex.jpg");
	detectAndDisplay(curr_frame, frame_gray, face_cascade, eyes_cascade);
#endif
    waitKey();
    return 0;
}