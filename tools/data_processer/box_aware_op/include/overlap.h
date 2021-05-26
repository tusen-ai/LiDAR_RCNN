#ifndef OVERLAP_H
#define OVERLAP_H

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace std;

struct Point{      //顶点
    float x, y;
};

struct Line{      //线
    Point a, b;
    float angle;//极角
    Line& operator= (Line l)
    {
        a.x = l.a.x;
        a.y = l.a.y;
        b.x = l.b.x;
        b.y = l.b.y;
        angle = l.angle;
        return *this;
    }
};

class OverlapChecker {
private:
    static const int MAX_SIZE = 16;
    static constexpr float EPS = 1e-5;
    int pn, dq[MAX_SIZE], top, bot;//数组模拟双端队列
    int n = 8;
    int pn_start = 8;
    Point p[MAX_SIZE];
    Line l[MAX_SIZE];

public:

    void clear_dq() {
        memset(dq, 0, sizeof(dq));
    }

    static int dblcmp(float k){                      //精度函数
        if (fabs(k) < EPS) return 0;
        return k > 0 ? 1 : -1;
    }

   static float multi(Point p0, Point p1, Point p2){                   //叉积
        return (p1.x-p0.x)*(p2.y-p0.y)-(p1.y-p0.y)*(p2.x-p0.x);
    }

    static bool cmp(const Line& l1, const Line& l2){
        int d = dblcmp(l1.angle-l2.angle);
        if (!d)
        return dblcmp(OverlapChecker::multi(l1.a, l2.a, l2.b)) > 0;
                    //大于0取半平面的左半，小于0取右半
        return d < 0;
    }

    void addLine(Line& l, float x1, float y1, float x2, float y2){
        l.a.x = x1;
        l.a.y = y1;
        l.b.x = x2;
        l.b.y = y2;
        l.angle = atan2(y2-y1, x2-x1);
    }

    void getIntersect(Line l1, Line l2, Point& p){
        float A1 = l1.b.y - l1.a.y;
        float B1 = l1.a.x - l1.b.x;
        float C1 = (l1.b.x - l1.a.x) * l1.a.y - (l1.b.y - l1.a.y) * l1.a.x;
        float A2 = l2.b.y - l2.a.y;
        float B2 = l2.a.x - l2.b.x;
        float C2 = (l2.b.x - l2.a.x) * l2.a.y - (l2.b.y - l2.a.y) * l2.a.x;
        p.x = (C2 * B1 - C1 * B2) / (A1 * B2 - A2 * B1);
        p.y = (C1 * A2 - C2 * A1) / (A1 * B2 - A2 * B1);
    }

    bool judge(Line l0, Line l1, Line l2){
        Point p;
        getIntersect(l1, l2, p);
        return dblcmp(multi(p, l0.a, l0.b)) > 0;
        //大于0，是p在向量l0.a->l0.b的左边，小于0是在右边，当p不在半平面l0内时，返回true
    }

    bool checkClockwise(Point p0, Point p1, Point p2) {         //判断是否点的顺序是顺时针
        return ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) > 0;
    }

    void HalfPlaneIntersect(){
        int i, j;
        sort(l, l+n, OverlapChecker::cmp); //极角排序
        for (i = 0, j = 0; i < n; i++)
            if (dblcmp(l[i].angle-l[j].angle) > 0)
                l[++j] = l[i];//排除极角相同（从了l[1]开始比较）
        int t = j + 1;//个数
        dq[0] = 0;//双端队列
        dq[1] = 1;//开始入队列两条直线
        top = 1;
        bot = 0;
        for (i = 2; i < t; i++)
        {
            while (top > bot && judge(l[i], l[dq[top]], l[dq[top-1]])) top--;
            while (top > bot && judge(l[i], l[dq[bot]], l[dq[bot+1]])) bot++;
            dq[++top] = i;
        }
        while (top > bot && judge(l[dq[bot]], l[dq[top]], l[dq[top-1]])) top--;
        while (top > bot && judge(l[dq[top]], l[dq[bot]], l[dq[bot+1]])) bot++;
        dq[++top] = dq[bot];
        for (pn = pn_start, i = bot; i < top; i++, pn++)
            getIntersect(l[dq[i+1]], l[dq[i]], p[pn]);//更新重复利用p数组
    }

    float getArea(int start, int end){
        if (end - start < 3) return 0;
        float area = 0;

        for (int i = start+1; i < end-1; i++)
            area += multi(p[start], p[i], p[i+1]);//利用p数组求面积
        if (area < 0) area = -area;
        return area/2;
    }

    void readRec(const py::EigenDRef<Eigen::MatrixXf> boxes, int i, int start){
        int k;
        for (k = 0; k < 4; ++k) {
            p[start+k].x = boxes(i, k*2);
            p[start+k].y = boxes(i, k*2+1);
        }
        bool tag = checkClockwise(p[start], p[start+1], p[start+2]);
        if (tag) reverse(p+start, p+start+4);
    }

    void overlap(const py::EigenDRef<Eigen::MatrixXf> boxes1,
         const py::EigenDRef<Eigen::MatrixXf> boxes2,
         py::EigenDRef<Eigen::MatrixXf> iou_matrix) {
        int k1 = boxes1.rows();
        int k2 = boxes2.rows();
        for (int j = 0; j < k2; ++j){
            readRec(boxes2, j, 0);
            float area2 = getArea(0, 4);
            for (int i = 0; i < k1; ++i) {
                clear_dq();
                readRec(boxes1, i, 4);
                for (int z = 0; z < 4; ++z){ //读入直线
                    addLine(l[z], p[z].x, p[z].y, p[(z+1)%4].x, p[(z+1)%4].y);
                    addLine(l[z+4], p[z+4].x, p[z+4].y, p[(z+1)%4+4].x, p[(z+1)%4+4].y);
                }
                // TODO: calculate the area outside the loop
                float area1 = getArea(4, 8);
                HalfPlaneIntersect();
                float iou = getArea(pn_start, pn);
                iou_matrix(i, j) = iou / (area1 + area2 - iou);
            }
        }
    }
};

void overlap(const py::EigenDRef<Eigen::MatrixXf> boxes1,
         const py::EigenDRef<Eigen::MatrixXf> boxes2,
         py::EigenDRef<Eigen::MatrixXf> iou_matrix);

#endif
