#ifndef POLYGON_OVERLAP_H
#define POLYGON_OVERLAP_H

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <pybind11/eigen.h>
#include <vector>
#include <iostream>

#define EPSILON 0.000001f

namespace py = pybind11;
using namespace std;



class Point2D {
    public:
     float x;
     float y;
     float angle; // for ordering

    Point2D(float x, float y) {
        this->x = x;
        this->y = y;
    }

    Point2D() {
    }

    Point2D(const Point2D& p) {
        this->x = p.x;
        this->y = p.y;
        this->angle = p.angle;
    }
};



class ConvexPolygon {
    public:
     vector<Point2D> corners;

    ConvexPolygon(vector<Point2D> corners) {
        this->corners = corners;
    }

    ConvexPolygon() {
    }

    float getArea() {
        if (this->corners.size() < 3) return 0;
        float area = 0;

//        cout << "--------------" << endl;
//        for (auto p: this->corners) {
//            cout << p.x << " " << p.y << endl;
//        }

        for (auto i = 1; i < this->corners.size() - 1; ++i) {
//            cout << "help" << this->corners.size() - 1 << endl;;
            area += cross_dot(this->corners[0], this->corners[i], this->corners[i+1]);
        }
        if (area < 0) area = -area;
        return area / 2;
    }

    static float cross_dot(const Point2D& p0, const Point2D& p1, const Point2D& p2){
        return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
    }

};




bool isEqual(float d1, float d2);

bool getIntersectionPoint(const Point2D& l1p1, const Point2D& l1p2,
                            const Point2D& l2p1, const Point2D& l2p2,
                            Point2D& output);

bool isPointInsidePoly(const Point2D& point, const ConvexPolygon& poly);

vector<Point2D> getIntersectionPoints(const Point2D& l1p1, const Point2D& l1p2,
                                    const ConvexPolygon& poly);


void addPoints(vector<Point2D> &pool, vector<Point2D> newpoints);

void orderByClockwise(vector<Point2D> &points);

ConvexPolygon getIntersectionOfPolygons(const ConvexPolygon& poly1,
                                        const ConvexPolygon& poly2);

ConvexPolygon readBox(const py::EigenDRef<Eigen::MatrixXf> boxes, int i);


void polygon_overlap(const py::EigenDRef<Eigen::MatrixXf> boxes1,
                     const py::EigenDRef<Eigen::MatrixXf> boxes2,
                     py::EigenDRef<Eigen::MatrixXf> iou_matrix);

#endif