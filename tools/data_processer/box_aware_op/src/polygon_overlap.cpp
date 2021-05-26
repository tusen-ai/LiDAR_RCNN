#include "polygon_overlap.h"

void polygon_overlap(const py::EigenDRef<Eigen::MatrixXf> boxes1,
                     const py::EigenDRef<Eigen::MatrixXf> boxes2,
                     py::EigenDRef<Eigen::MatrixXf> iou_matrix) {
      int k1 = boxes1.rows();
      int k2 = boxes2.rows();

      ConvexPolygon poly1, poly2, intersectPoly;

      for (int j = 0; j < k2; ++j) {
          poly2 = readBox(boxes2, j);
          float area2 = poly2.getArea();
          for (int i = 0; i < k1; ++i) {
              poly1 = readBox(boxes1, i);
              float area1 = poly1.getArea();
              intersectPoly = getIntersectionOfPolygons(poly1, poly2);
              float intersectArea = intersectPoly.getArea();
              iou_matrix(i, j) = intersectArea / (area1 + area2 - intersectArea);
          }
      }
}

bool isEqual(float d1, float d2) {
    return fabs((d1 - d2) / min(d1, d2)) < EPSILON;
}

bool getIntersectionPoint(const Point2D& l1p1, const Point2D& l1p2,
                            const Point2D& l2p1, const Point2D& l2p2,
                            Point2D& output) {
    float A1 = l1p2.y - l1p1.y;
    float B1 = l1p1.x - l1p2.x;
    float C1 = A1 * l1p1.x + B1 * l1p1.y;

    float A2 = l2p2.y - l2p1.y;
    float B2 = l2p1.x - l2p2.x;
    float C2 = A2 * l2p1.x + B2 * l2p1.y;

    // lines are parallel
    float det = A1 * B2 - A2 * B1;

    if (isEqual(det, 0.0)) {
        return false;
    }
    else {
        float x = (B2 * C1 - B1 * C2) / det;
        float y = (A1 * C2 - A2 * C1) / det;
        bool online1 = ((min(l1p1.x, l1p2.x) < x || isEqual(min(l1p1.x, l1p2.x), x))
            && (max(l1p1.x, l1p2.x) > x || isEqual(max(l1p1.x, l1p2.x), x))
            && (min(l1p1.y, l1p2.y) < y || isEqual(min(l1p1.y, l1p2.y), y))
            && (max(l1p1.y, l1p2.y) > y || isEqual(max(l1p1.y, l1p2.y), y))
            );
        bool online2 = ((min(l2p1.x, l2p2.x) < x || isEqual(min(l2p1.x, l2p2.x), x))
            && (max(l2p1.x, l2p2.x) > x || isEqual(max(l2p1.x, l2p2.x), x))
            && (min(l2p1.y, l2p2.y) < y || isEqual(min(l2p1.y, l2p2.y), y))
            && (max(l2p1.y, l2p2.y) > y || isEqual(max(l2p1.y, l2p2.y), y))
            );
        if (online1 && online2) {
            output.x = x;
            output.y = y;
            return true;
        }
    }
    return false;
}

bool isPointInsidePoly(const Point2D& point, const ConvexPolygon& poly) {
    int i, j;
    bool result = false;
    // judge whether is the same point
    for (auto p: poly.corners) {
        if (isEqual(p.x, point.x) && (isEqual(p.y, point.y))) {
            return true;
        }
    }
    for (i = 0, j = poly.corners.size() - 1; i < poly.corners.size(); j = i++) {
        if ((poly.corners[i].y > point.y) != (poly.corners[j].y > point.y) &&
            (point.x < (poly.corners[j].x - poly.corners[i].x) * (point.y - poly.corners[i].y) / (poly.corners[j].y - poly.corners[i].y) + poly.corners[i].x)) {
            result = !result;
        }
    }
    return result;
}

vector<Point2D> getIntersectionPoints(const Point2D& l1p1, const Point2D& l1p2,
                                    const ConvexPolygon& poly) {
    vector<Point2D> intersectionPoints;

    for (int i = 0; i < poly.corners.size(); ++i) {
        int next = i + 1 == poly.corners.size() ? 0 : i + 1;
        Point2D intersectPoint = Point2D();
        bool intersect = getIntersectionPoint(l1p1, l1p2, poly.corners[i], poly.corners[next], intersectPoint);

        if (intersect) intersectionPoints.push_back(intersectPoint);
    }
    return intersectionPoints;
}


void addPoints(vector<Point2D> &pool, vector<Point2D> newpoints) {
    for(auto np : newpoints) {
        bool found = false;
        for (auto p : pool) {
            if (isEqual(np.x, p.x) && isEqual(np.y, p.y)) {
                found = true;
                break;
            }
        }
        if (!found) pool.push_back(Point2D(np));
    }
}

void orderByClockwise(vector<Point2D> &points) {
    float mx = 0;
    float my = 0;

    for (auto p : points) {
        mx += p.x;
        my += p.y;
    }

    mx /= (float)points.size();
    my /= (float)points.size();

    for (auto &p: points) {
        p.angle = atan2(p.y - my, p.x - mx);
    }

    sort(points.begin( ), points.end( ), [ ]( const Point2D& lhs, const Point2D& rhs)
    {
        return lhs.angle < rhs.angle;
    });
}

ConvexPolygon getIntersectionOfPolygons(const ConvexPolygon& poly1, const ConvexPolygon& poly2) {
    vector<Point2D> clippedCorners;

    // Add the corners of poly1 which are inside poly2
    for (auto p : poly1.corners) {
        if (isPointInsidePoly(p, poly2)) {
            vector<Point2D> newpoints = {Point2D(p)};
            addPoints(clippedCorners, newpoints);
        }
    }
    // Add the corners of poly2 which are inside poly1
    for (auto p : poly2.corners) {
        if (isPointInsidePoly(p, poly1)) {
            vector<Point2D> newpoints = {Point2D(p)};
            addPoints(clippedCorners, newpoints);
        }
    }
    // Add the intersection points
    for (int i = 0, next = 1; i < poly1.corners.size(); ++i, next = ((i + 1) == poly1.corners.size() ? 0 : i + 1)) {
        addPoints(clippedCorners, getIntersectionPoints(poly1.corners[i], poly1.corners[next], poly2));
    }

    // order by clockwise
    orderByClockwise(clippedCorners);

    return ConvexPolygon(clippedCorners);
}

ConvexPolygon readBox(const py::EigenDRef<Eigen::MatrixXf> boxes, int i) {
    vector<Point2D> corners;

    for (int k = 0; k < 4; ++k) {
        corners.push_back(Point2D(boxes(i, k*2), boxes(i, k*2+1)));
    }
    orderByClockwise(corners);
    return ConvexPolygon(corners);
}
