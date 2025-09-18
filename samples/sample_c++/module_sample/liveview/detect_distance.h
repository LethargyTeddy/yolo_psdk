#ifndef DETECT_DISTANCE_H
#define DETECT_DISTANCE_H

#include<iostream>
#include "opencv2/opencv.hpp"
// #include <opencv2/core/mat.hpp>
struct cameraIntrinsics{
    double fx;
    double fy;
    double cx;
    double cy;
}; //相机内参

struct F_Vector3f {
    float pitch; /*!< pitch */
    float roll; /*!< roll */
    float yaw; /*!< yaw */
}; //云台欧拉角

struct D_Vector3f {
    double x; /*!<  */
    double y; /*!<  */
    double z; /*!<  */
}; //wgs84坐标系/ecef坐标系

struct D_HomePointInfo {
    double latitude;  /*!< unit: rad */
    double longitude; /*!< unit: rad */
};  // 参考点WGS-84坐标（单位为弧度）

struct BBoxCenter {
    int x; //bounding box center x
    int y; //bounding box center y
};

class DetectDistance {
public:
DetectDistance(const double fx,const double fy,const double cx,const double cy);
DetectDistance(const cv::Mat& cameraIntrinsicMatrix);
~DetectDistance();
void CalculateDistance();
void UploadGimbalAngles(F_Vector3f& gimbalAngles);
void UploadOriginGPS(const double& latitude,const double& longitude);
void UploadCameraHeight(const float& cameraHeight);
void UploadbboxCenter(const int bboxCenterX ,const int bboxCenterY);
void HoverPoint(double& latitude,double& longitude);
bool BBoxIsEmpty();
D_Vector3f nedToEcef(double N, double E, double D, double lat0, double lon0, double alt0);// 计算ECEF坐标
D_Vector3f ecefToWgs84(double x, double y, double z);// 计算WGS-84坐标
D_Vector3f wgs84ToEcef(double lat, double lon, double alt);
D_Vector3f outputWgs84();
private:
cameraIntrinsics camInr;
double camHeight;
F_Vector3f gimbalAngles;
D_HomePointInfo originGPS;
D_Vector3f wgs84;
D_Vector3f ecef;
BBoxCenter bboxCenter = {0,0};
// WGS-84参数
const double a = 6378137.0;              // 地球长半轴
// const double f = 1.0 / 298.257223563;    // 扁率
const double e2 = 6.69437999014e-3;         // 偏心率平方
};

#endif