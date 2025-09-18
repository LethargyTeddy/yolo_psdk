#include "detect_distance.h"
#include <cmath>

DetectDistance::DetectDistance(const double fx,const double fy,const double cx,const double cy)
: camInr{fx,fy,cx,cy}
{}
DetectDistance::DetectDistance(const cv::Mat& cameraIntrinsicMatrix)
: camInr{cameraIntrinsicMatrix.at<double>(0,0),
cameraIntrinsicMatrix.at<double>(1,1),
cameraIntrinsicMatrix.at<double>(0,2),
cameraIntrinsicMatrix.at<double>(1,2)}
{}
DetectDistance::~DetectDistance(){}
void DetectDistance::CalculateDistance()
{
    double camera_x = (this->bboxCenter.x - this->camInr.cx) * this->camHeight / this->camInr.fx;
    double camera_y = (this->bboxCenter.y - this->camInr.cy) * this->camHeight / this->camInr.fy;
    int theta = this->gimbalAngles.yaw + 90;//yaw + 90
    if (theta > 180) theta -= 360;
    double local_x = camera_x * std::cos(theta) + camera_y * std::sin(theta);//N北：x
    double local_y = -camera_x * std::sin(theta) + camera_y * std::cos(theta);//E东：y
    this->ecef = this->nedToEcef(local_x, local_y, -this->camHeight, this->originGPS.latitude, this->originGPS.longitude, this->camHeight);
    this->wgs84 = this->ecefToWgs84(this->ecef.x, this->ecef.y, this->ecef.z);

    return;
}
D_Vector3f DetectDistance::nedToEcef(double N, double E, double D, double lat0, double lon0, double alt0){// 计算ECEF坐标
        // 参考点的ECEF坐标
    D_Vector3f ecefRef = this->wgs84ToEcef(lat0, lon0, alt0);

    // 旋转矩阵
    double sinLat = std::sin(lat0);
    double cosLat = std::cos(lat0);
    double sinLon = std::sin(lon0);
    double cosLon = std::cos(lon0);

    // NED 到 ECEF 的转换
    D_Vector3f ecef;
    ecef.x = ecefRef.x - sinLat * cosLon * N - sinLon * E - cosLat * cosLon * D;
    ecef.y = ecefRef.y - sinLat * sinLon * N + cosLon * E - cosLat * sinLon * D;
    ecef.z = ecefRef.z + cosLat * N - sinLat * D;

    return ecef;
}
D_Vector3f DetectDistance::ecefToWgs84(double x, double y, double z){// 计算WGS-84坐标
    double p = std::sqrt(x * x + y * y);
    double theta = std::atan2(z * this->a, p * (1 - this->e2));
    double sinTheta = std::sin(theta);
    double cosTheta = std::cos(theta);

    D_Vector3f wgs84;//wgs84.x = lat ,wgs84.y = lon ,wgs84.z = alt
    wgs84.x = std::atan2(z + this->e2 * (1 - this->e2) * this->a * sinTheta * sinTheta * sinTheta,
                      p - this->e2 * this->a * cosTheta * cosTheta * cosTheta);
    wgs84.y = std::atan2(y, x);

    double N = this->a / std::sqrt(1 - this->e2 * std::sin(wgs84.x) * std::sin(wgs84.x));
    wgs84.z = p / std::cos(wgs84.x) - N;

    return wgs84;
}
D_Vector3f DetectDistance::wgs84ToEcef(double lat, double lon, double alt){
    double sinLat = std::sin(lat);
    double cosLat = std::cos(lat);
    double sinLon = std::sin(lon);
    double cosLon = std::cos(lon);

    double N = this->a / std::sqrt(1 - this->e2 * sinLat * sinLat);
    D_Vector3f ecef;
    ecef.x = (N + alt) * cosLat * cosLon;
    ecef.y = (N + alt) * cosLat * sinLon;
    ecef.z = (N * (1 - this->e2) + alt) * sinLat;
    return ecef;
}
D_Vector3f DetectDistance::outputWgs84(){
    return this->wgs84;
}
bool DetectDistance::BBoxIsEmpty(){
    if (this->bboxCenter.x||this->bboxCenter.y) return false;
    return true;
}
void DetectDistance::UploadGimbalAngles(F_Vector3f& gimbalAngles){
    this->gimbalAngles = gimbalAngles;
}
void DetectDistance::UploadOriginGPS(const double& latitude,const double& longitude){
    this->originGPS.latitude = latitude;
    this->originGPS.longitude = longitude;
}
void DetectDistance::UploadCameraHeight(const float& cameraHeight){
    this->camHeight = cameraHeight;
}
void DetectDistance::UploadbboxCenter(const int bboxCenterX ,const int bboxCenterY){
    this->bboxCenter.x = bboxCenterX;
    this->bboxCenter.y = bboxCenterY;
}
void DetectDistance::HoverPoint(double& latitude,double& longitude){
    latitude = this->wgs84.x;//wgs84.x = lat ,wgs84.y = lon ,wgs84.z = alt
    longitude = this->wgs84.y;
}