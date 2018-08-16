# 빠른 얼굴 검출을 위한 **IDEA** 3가지

---
##1. Integral Image##

- 이미지의 밝기차이를 지속적으로 계산하기엔 시간이 많이 걸린다.

- 아래와 같이 Integral Image 처리 후 미리 계산된 밝기 합을 사용한다면 계산 시간을 크게 줄 일 수가 있다.

>  ![ ](/image/intgral.jpg)

- 이미지 Mat 객체를 받아 Integral Image 객체 생성

> ```c++
void integralImage(Mat &src, Mat &intg) {
    int calcVal = 0;
    intg = Mat::zeros(src.rows + 1, src.cols + 1, CV_32SC1);
    uchar* src_d = src.data;
    int* intg_d = intg.ptr<int>(1, 1);
    for (int x = 0; x < src.rows; x++) {
        for (int y = 0; y < src.cols; y++) {
            calcVal = *(src_d);
            *(intg_d) =
                (int)calcVal
                + (int)*(intg_d - intg.step1())
                + (int)*(intg_d - 1)
                - (int)*(intg_d - intg.step1() - 1);
            intg_d++;
            src_d++;
        }
        intg_d++;
    }
}
```



##2. Haar-like Feature##

- 검정영역의 밝기와 흰 영역의 밝기 차이에서 임계값 이상은 것을 찾는것 

> ![ ](/image/haar_feature.jpg)
 
-  Mat 객체와 밝기 지정 범위를 받아 밝기 차이 값을 반환

> ```c++
 int getIntegralSum(Mat &intg, int col, int row, int width, int height) {
    int* val = intg.ptr<int>(0);
    int step = (int)intg.step1();
    return *(val + (step * height) + width) + *(val + (step * row) + col)
        - *(val + (step * height) + col) - *(val + (step * row) + width);
}
```

>```c++
int haarLikeValue(Mat &intg, int col, int row, int width, int height) {
    if ((width % 2) != 0) return 0;
    return getIntegralSum(intg, col, row, col + (width / 2), row + height) -
        getIntegralSum(intg, col + (width / 2), row, col + width, row + height);
}
```

- 예제 실행 결과

[01_haar_ex.cpp](https://github.com/yellowdo/opencv_haar/blob/master/01_haar_ex.cpp "01_haar_ex.cpp")

>![ ](/image/intg.jpg)


##3. LBP , MCT##
-  LBP (Local Binary Pattern)
	- 이미지의 Texture(질감)표현 및 얼굴 인식 등에 활용
	영상의 밝기 변화에 강인한 특징을 가짐
	단순한 밝기의 변화는 LBP 연산에 영향을 미치지 않음

	- 구현 방법
	3x3 셀 내에서 중심에 위치하는 픽셀과 이웃하는 8개의 픽셀들과 서로 값을 비교
	이웃하는 픽셀 값이 중심 값보다 크면 1, 작으면 0으로 셋팅
	순서대로 나열 => 모든 픽셀에 대하여 적용 후 히스토그램(픽셀 값의 히스토그램이 아니라 LBP 값들의 히스토그램을 만드는 것: 하나의 영상의 질감을 __256__개의 숫자로 표현)

>```c++
#define COMP_TH(X, COMP, VAL) (X >= COMP ? VAL : 0)
class Parallel_LBP : public ParallelLoopBody {
private:
    Mat & src;
    int* dst;
public:
    Parallel_LBP(Mat &_src, int* _dst) : src(_src), dst(_dst) {}
    virtual void operator()(const Range &r) const {
        int row = src.rows, col = src.cols, step = (int)src.step1();
        uchar* ptr = src.ptr<uchar>(0, 0);
        for (int i = r.start; i < r.end; i++) { // row 병렬처리
            int tot, comp;
            for (int j = 1; j < col - 1; j++) {
                comp = *(ptr + (step * i) + j);
                tot = COMP_TH(*(ptr + (step * i) - 1 + j), comp, 1);
                tot += COMP_TH(*(ptr + (step * i) + step - 1 + j), comp, 2);
                tot += COMP_TH(*(ptr + (step * i) + step + j), comp, 4);
                tot += COMP_TH(*(ptr + (step * i) + step + 1 + j), comp, 8);
                tot += COMP_TH(*(ptr + (step * i) + 1 + j), comp, 16);
                tot += COMP_TH(*(ptr + (step * i) - step + 1 + j), comp, 32);
                tot += COMP_TH(*(ptr + (step * i) - step + j), comp, 64);
                tot += COMP_TH(*(ptr + (step * i) - step - 1 + j), comp, 128);
                *(dst + ((row - 2) * (i - 1)) + (j - 1)) = tot;
            }
        }
    }
};
```

 - 예제 실행 결과
[02_LbpMct_ex.cpp](https://github.com/yellowdo/opencv_haar/blob/master/02_LbpMct_ex.cpp "02_LbpMct_ex.cpp")
> ![ ](/image/haar_feature.jpg)

- Uniform LBP
	- 어떤 패턴들은 좀 더 영상 내에서 자주 발견되는 반면 어떤 패턴들은 드물게 발견되므로 특성의 개수를 줄일 수 있는 방법이 제안 
	0 => 1 또는 1=>0으로의 변화가 2번 이내인 패턴: Uniform 패턴 (각각을 Binning)
	0 => 1 또는 1=>0으로의 변화가 3번 이상인 패턴: Non-Uniform 패턴 (하나의 Binning으로 처리)
	- 256 Binning => __59__ Binning (58: Uniform, 1: Non-Uniform)

- Rotation-invariant LBP
	- LBP / Uniform LBP: 이미지 회전에 취약
	밝기 변화와 회전에도 강건한 특징을 만들기 위해 제안
> ![ ](/image/riLBP.jpg)
	- 256 Bin => 59 Bin (Uniform LBP) => __10__ Bin

>```c++
static void invariant_rotaion_lbp(int *lbp, int size_lbp, float *hist) {
    static const int lookup[7][8] = {
        { 1, 2, 4, 8, 16, 32, 64, 128 },
        { 3, 6, 12, 24, 48, 96, 192, 129 },
        { 7, 14, 28, 56, 112, 224, 193, 131 },
        { 15, 30, 60, 120, 240, 225, 195, 135 },
        { 31, 62, 124, 248, 241, 227, 199, 143 },
        { 63, 126, 252, 249, 243, 231, 207, 159 },
        { 127, 254, 253, 251, 247, 239, 223, 191 } };
    int bin_count[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    bool non_uniform;
    for (int i = 0; i < size_lbp; i++) {
        if (lbp[i] == 0) {
            bin_count[0]++;
        }
        else if (lbp[i] == 255) {
            bin_count[8]++;
        }
        else {
            for (int j = 0; j < 8; j++) {
                non_uniform = false;
                if (lbp[i] == lookup[0][j]) {
                    bin_count[1]++;
                    break;
                }
                else if (lbp[i] == lookup[1][j]) {
                    bin_count[2]++;
                    break;
                }
                else if (lbp[i] == lookup[2][j]) {
                    bin_count[3]++;
                    break;
                }
                else if (lbp[i] == lookup[3][j]) {
                    bin_count[4]++;
                    break;
                }
                else if (lbp[i] == lookup[4][j]) {
                    bin_count[5]++;
                    break;
                }
                else if (lbp[i] == lookup[5][j]) {
                    bin_count[6]++;
                    break;
                }
                else if (lbp[i] == lookup[6][j]) {
                    bin_count[7]++;
                    break;
                }
                else non_uniform = true;
            }
            // 0, 255, Uniform에도 속하지 않는 값
            if (non_uniform) {
                bin_count[9]++;
            }
        }
    }
    for (int k = 0; k < 10; k++) {
        hist[k] = (float)(bin_count[k] / 324.f);
    }
}
```

- 예제 실행 결과
[03_face.cpp](https://github.com/yellowdo/opencv_haar/blob/master/03_face.cpp "03_face.cpp")
> ![ ](/image/compare.jpg)


##4. Cascade Adaboost##

- Boosting
	반복 학습 방법
	
- Adaptive + Boosting
	간단한 weak classifier들이 상호보완 되도록 순차적으로 학습진행하여 이들을 조합 하여 strong classifier의 성능을 향상 시킴.
	weak classifier들을 학습시 먼저 학습된 분류기의 오류 정보를 다음 분류기의 학습시 사용하여 단점을 보완.
	이전 분류기 오류의 가중치를 adaptive 하게 변경해가며 잘못 분류되는 데이터에 더 집중하여  잘 학습하고 분류할수있도록 함.

- 요약
> ![ ](/image/adaboost.jpg)

- 예제 실행 결과
[04_face_dection_lib.cpp](https://github.com/yellowdo/opencv_haar/blob/master/04_face_dection_lib.cpp "04_face_dection_lib.cpp")
> ![ ](/image/face_dectect.jpg)