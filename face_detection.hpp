#pragma once
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

#define COMP_TH(X, COMP, VAL) (X >= COMP ? VAL : 0)

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

int getIntegralSum(Mat &intg, int col, int row, int width, int height) {
    int* val = intg.ptr<int>(0);
    int step = (int)intg.step1();
    return *(val + (step * height) + width) + *(val + (step * row) + col)
        - *(val + (step * height) + col) - *(val + (step * row) + width);
}

int haarLikeValue(Mat &intg, int col, int row, int width, int height) {
    if ((width % 2) != 0) return 0;
    return getIntegralSum(intg, col, row, col + (width / 2), row + height) -
        getIntegralSum(intg, col + (width / 2), row, col + width, row + height);
}

class Parallel_LBP_MAT : public ParallelLoopBody {
private:
    Mat & src, &dst;
public:
    Parallel_LBP_MAT(Mat &_src, Mat& _dst) : src(_src), dst(_dst) {
        this->dst = Mat::zeros(src.rows, src.cols, src.type());
    }
    virtual void operator()(const Range &r) const {
        int row = src.rows, col = src.cols, step = (int)src.step1();
        uchar* ptr = src.ptr<uchar>(0, 0);
        uchar* ptr_dst = dst.ptr<uchar>(0, 0);
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
                *(ptr_dst + (step * i) + j) = tot;
            }
        }
    }
};

class Parallel_MCT_MAT : public ParallelLoopBody {
private:
    Mat & src, &dst;
public:
    Parallel_MCT_MAT(Mat &_src, Mat& _dst) : src(_src), dst(_dst) {
        this->dst = Mat::zeros(src.rows, src.cols, src.type());
    }
    virtual void operator()(const Range &r) const {
        int row = src.rows, col = src.cols, step = (int)src.step1();
        uchar* ptr = src.ptr<uchar>(0, 0);
        uchar* ptr_dst = dst.ptr<uchar>(0, 0);
        for (int i = r.start; i < r.end; i++) { // row 병렬처리
            uchar val[9];
            int tot, input;
            for (int j = 1; j < col - 1; j++) {
                val[0] = *(ptr + (step * i) - step - 1 + j);
                val[1] = *(ptr + (step * i) - step + j);
                val[2] = *(ptr + (step * i) - step + 1 + j);
                val[3] = *(ptr + (step * i) + 1 + j);
                val[4] = *(ptr + (step * i) + step + 1 + j);
                val[5] = *(ptr + (step * i) + step + j);
                val[6] = *(ptr + (step * i) + step - 1 + j);
                val[7] = *(ptr + (step * i) - 1 + j);
                val[8] = *(ptr + (step * i) + j);
                tot = val[0] + val[1] + val[2] + val[3] + val[4] +
                    val[5] + val[6] + val[7] + val[8];

                input = COMP_TH((val[0] << 3) + val[0], tot, 1);
                input += COMP_TH((val[1] << 3) + val[1], tot, 2);
                input += COMP_TH((val[2] << 3) + val[2], tot, 4);
                input += COMP_TH((val[3] << 3) + val[3], tot, 8);
                input += COMP_TH((val[4] << 3) + val[4], tot, 16);
                input += COMP_TH((val[5] << 3) + val[5], tot, 32);
                input += COMP_TH((val[6] << 3) + val[6], tot, 64);
                input += COMP_TH((val[7] << 3) + val[7], tot, 128);
                *(ptr_dst + (step * i) + j) = input;
            }
        }
    }
};

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

class mat_processor {
public:
    virtual void print() = 0;
    virtual float run() = 0;
};

class CompareLBP : public mat_processor {
private:
    Mat & src, &dst;
    String src_name, dst_name;
    float *bin, *bin_dst;
    int blockSize;
    int blockCount;
    float diff;
    const int binArray = 10;

public:
    CompareLBP(Mat &_src, String s_name, Mat &_dst, String d_name)
        : src(_src), src_name(s_name), dst(_dst), dst_name(d_name), blockSize(20), blockCount(10), diff(0.f) {
    }
    void print() {
        cout << src_name << " vs " << dst_name << " = " << diff << endl;
    }

    float run() {
        diff = 0.f;
        setMatrix(src);
        setMatrix(dst);
        bin = (float*)malloc(sizeof(float) * blockCount * blockCount * binArray);
        bin_dst = (float*)malloc(sizeof(float) * blockCount * blockCount * binArray);
        for (int i = 0; i < blockCount; i++) {
            for (int j = 0; j < blockCount; j++) {
                Mat cropImg = src(Rect(i * blockSize, j * blockSize, blockSize, blockSize));
                Mat cropImg_dst = dst(Rect(i * blockSize, j * blockSize, blockSize, blockSize));
                int *buf = (int*)malloc(sizeof(int) * (cropImg.rows - 2) * (cropImg.cols - 2));

                parallel_for_(Range(1, cropImg.rows - 1), Parallel_LBP(cropImg, buf));
                invariant_rotaion_lbp(
                    buf, (cropImg.rows - 2) * (cropImg.cols - 2),
                    &bin[(j * blockCount * blockCount) + (i * binArray)]);

                parallel_for_(Range(1, cropImg_dst.rows - 1), Parallel_LBP(cropImg_dst, buf));
                invariant_rotaion_lbp(
                    buf, (cropImg_dst.rows - 2) * (cropImg_dst.cols - 2),
                    &bin_dst[(j * blockCount * blockCount) + (i * binArray)]);

                free(buf);
            }
        }
        for (int k = 0; k < blockCount * blockCount * binArray; k++) {
            diff += (bin[k] - bin_dst[k]) * (bin[k] - bin_dst[k]);
        }
        free(bin);
        free(bin_dst);
        return diff;
    }
    void setMatrix(Mat &src) {
        if (src.channels() > 1)
            cvtColor(src, src, CV_RGB2GRAY);
        resize(src, src, Size((blockCount * blockSize), (blockCount * blockSize)));
    }
    void setParam1(Mat &_src, String _name) {
        src = _src;
        src_name = _name;
        setMatrix(src);
    }
    void setParam2(Mat &_dst, String _name) {
        dst = _dst;
        dst_name = _name;
        setMatrix(dst);
    }
};

class ContainerProcess {
private:
    vector<mat_processor*> v;
    vector<mat_processor*>::iterator it;
public:
    void add(mat_processor* proc) {
        v.push_back(proc);
    }
    void del() {
        v.pop_back();
    }
    void run() {
        it = v.begin();
        for (int i = 0; it != v.end(); it++, i++) {
            v[i]->run();
            v[i]->print();
        }
    }
};