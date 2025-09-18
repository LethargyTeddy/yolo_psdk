/**
 ********************************************************************
 * @file    test_liveview_entry.cpp
 * @brief
 *
 * @copyright (c) 2021 DJI. All rights reserved.
 *
 * All information contained herein is, and remains, the property of DJI.
 * The intellectual and technical concepts contained herein are proprietary
 * to DJI and may be covered by U.S. and foreign patents, patents in process,
 * and protected by trade secret or copyright law.  Dissemination of this
 * information, including but not limited to data and other proprietary
 * material(s) incorporated within the information, in any form, is strictly
 * prohibited without the express written consent of DJI.
 *
 * If you receive this source code without DJI’s authorization, you may not
 * further disseminate the information, and you must immediately remove the
 * source code and notify DJI of its removal. DJI reserves the right to pursue
 * legal actions against you for any loss(es) or damage(s) caused by your
 * failure to do so.
 *
 *********************************************************************
 */

/* Includes ------------------------------------------------------------------*/
#include <cstdio>
#include <iostream>
#include <dji_logger.h>
#include <opencv2/core/mat.hpp>
#include <string>
#include "dji_typedef.h"
#include "test_liveview_entry.hpp"
#include "test_liveview.hpp"

// #include <termios.h>
// #include <utils/util_misc.h>
// #include <utils/util_file.h>
// #include <utils/cJSON.h>
#include <dji_aircraft_info.h>
// #include "test_flight_controller_command_flying.h"
#include "dji_flight_controller.h"
// #include "dji_logger.h"
#include "dji_fc_subscription.h"
#include "detect_distance.h"
// #include "test_interest_point.h"
#include "dji_interest_point.h"
#include "cmath"

#include <stdexcept>
// #include "test_gimbal_entry.hpp"
// #include <gimbal_manager/test_gimbal_manager.h>
// #include "dji_gimbal.h"
#include "dji_gimbal_manager.h"

// #include "chrono"
// #include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include "BYTETracker.h"

const std::vector<std::string> CLASS_NAMES = {
    "apron"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189}};

#ifdef OPEN_CV_INSTALLED

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../sample_c/module_sample/utils/util_misc.h"

#include <sys/stat.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <iostream>
#include <cerrno>


using namespace cv;
#endif
using namespace std;

/* Private constants ---------------------------------------------------------*/
#define DJI_TEST_COMMAND_FLYING_TASK_STACK_SIZE                          2048
#define DJI_TEST_COMMAND_FLYING_CTRL_FREQ                                50
#define DJI_TEST_COMMAND_FLYING_GO_HOME_ALTITUDE                         50
#define DJI_TEST_COMMAND_FLYING_CONTROL_SPEED_DEFAULT                    0.1
#define DJI_TEST_COMMAND_FLYING_RC_LOST_ACTION_STR_MAX_LEN               32
#define DJI_TEST_COMMAND_FLYING_CONFIG_DIR_PATH_LEN_MAX                  (256)
#define boundmax                                                         15
#define boundmin                                                         -15
#define fx                                                               2.827758455738353e+03
#define fy                                                               2.828931598626265e+03
#define cx                                                               2.000460921574551e+03
#define cy                                                               1.497583406990669e+03

/* Private types -------------------------------------------------------------*/

/* Private values -------------------------------------------------------------*/
static T_DjiTaskHandle s_commandFlyingTaskHandle;
static T_DjiTaskHandle s_statusDisplayTaskHandle;
static T_DjiFlightControllerJoystickCommand s_flyingCommand = {0};
static uint16_t s_inputFlag = 0;
static dji_f32_t s_flyingSpeed = DJI_TEST_COMMAND_FLYING_CONTROL_SPEED_DEFAULT;
static uint16_t s_goHomeAltitude = DJI_TEST_COMMAND_FLYING_GO_HOME_ALTITUDE;
static char s_rcLostActionString[DJI_TEST_COMMAND_FLYING_RC_LOST_ACTION_STR_MAX_LEN] = {0};
static T_DjiFlightControllerHomeLocation s_homeLocation = {0};
static T_DjiFcSubscriptionGpsPosition s_gpsPosition = {0};
static T_DjiFcSubscriptionPositionVO s_positionVo = {0};
static bool isFirstUpdateConfig = false;
static bool isCommandFlyingTaskStart = false;
static uint32_t s_statusDisplayTaskCnt = 0;
static T_DjiFcSubscriptionSingleBatteryInfo singleBatteryInfo1 = {0};
static T_DjiFcSubscriptionSingleBatteryInfo singleBatteryInfo2 = {0};
// const char *classNames[] = {"background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
//                             "boat", "traffic light",
//                             "fire hydrant", "background", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
//                             "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "background", "backpack",
//                             "umbrella", "background", "background", "handbag", "tie", "suitcase", "frisbee", "skis",
//                             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
//                             "surfboard", "tennis racket",
static YOLOv8* yolov8;
static BYTETracker* tracker;
static DetectDistance* detdistance;
// static cv::Mat * pimage;
//                             "bottle", "background", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
//                             "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
//                             "cake", "chair", "couch", "potted plant", "bed", "background", "dining table", "background",
//                             "background", "toilet", "background", "tv", "laptop", "mouse", "remote", "keyboard",
//                             "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "background", "book",
//                             "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

std::string outputFolder = "/home/nvidia/Desktop/yolo_psdk/output/";
const size_t inWidth = 320;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float) inHeight;
//static int32_t s_demoIndex = -1;
char curFileDirPath[DJI_FILE_PATH_SIZE_MAX];
char tempFileDirPath[DJI_FILE_PATH_SIZE_MAX];
char prototxtFileDirPath[DJI_FILE_PATH_SIZE_MAX];
char weightsFileDirPath[DJI_FILE_PATH_SIZE_MAX];
std::chrono::time_point<std::chrono::steady_clock> timestamp = std::chrono::steady_clock::now();
// long int timestamp = 0;
int num = 0;
int img_count = 0;
/* Private functions declaration ---------------------------------------------*/
static void *DjiUser_FlightControllerCommandFlyingTask(void *arg);
static void *DjiUser_FlightControllerStatusDisplayTask(void *arg);
static void DjiUser_ShowFlightStatusByOpenCV(void);
static void Custom_Flight(void);
static void Show_Dji_POI_Error(void);
static void DjiUser_FlightControllerVelocityAndYawRateCtrl(T_DjiFlightControllerJoystickCommand command);
// static int DjiUser_ScanKeyboardInput(void);
static T_DjiReturnCode
DjiUser_FlightCtrlJoystickCtrlAuthSwitchEventCb(T_DjiFlightControllerJoystickCtrlAuthorityEventInfo eventData);
static T_DjiVector3f DjiUser_FlightControlGetValueOfQuaternion(void);
static T_DjiVector3f DjiUser_FlightControlGetValueOfGimbalAngles(void);
static T_DjiFcSubscriptionGpsPosition DjiUser_FlightControlGetValueOfGpsPosition(void);
static T_DjiFcSubscriptionAltitudeOfHomePoint DjiUser_FlightControlGetValueOfRelativeHeight(void);
static T_DjiFcSubscriptionHomePointInfo DjiUser_FlightControlGetValueOfHomepointInfo(void);
static T_DjiFcSubscriptionPositionVO DjiUser_FlightControlGetValueOfPositionVo(void);
static T_DjiFcSubscriptionControlDevice DjiUser_FlightControlGetValueOfControlDevice(void);
static T_DjiFcSubscriptionSingleBatteryInfo DjiUser_FlightControlGetValueOfBattery1(void);
static T_DjiFcSubscriptionSingleBatteryInfo DjiUser_FlightControlGetValueOfBattery2(void);
// static T_DjiReturnCode DjiUser_FlightControlUpdateConfig(void);

static void DjiUser_ShowRgbImageCallback(CameraRGBImage img, void *userData);
static void TestImageCallback(CameraRGBImage img, void *userData);
static T_DjiReturnCode DjiUser_InterestPointMissionStateCallback(T_DjiInterestPointMissionState missionState);
static T_DjiReturnCode DjiUser_GetCurrentFileDirPath(const char *filePath, uint32_t pathBufferSize, char *dirPath);


// Ensure the std::filesystem namespace is used
// namespace fs = std::filesystem;

// Function to get formatted current time as string
std::string getCurrentTimeString() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}

bool createDirectory(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == 0) {
        return true;
    } else if (errno == EEXIST) {
        // Directory already exists
        return true;
    } else {
        // Some other error occurred
        perror("mkdir");
        return false;
    }
}

/* Exported functions definition ---------------------------------------------*/
void DjiUser_RunCameraStreamViewSample()
{
    char cameraIndexChar = 0;
    char demoIndexChar = 0;
    char isQuit = 0;
    CameraRGBImage camImg;
    // char fpvName[] = "FPV_CAM";
    char mainName[] = "/home/nvidia/Desktop/yolo_psdk/detect/airport.engine";
    // char viceName[] = "VICE_CAM";
    // char topName[] = "TOP_CAM";
    T_DjiReturnCode returnCode;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    LiveviewSample *liveviewSample;
    try {
        liveviewSample = new LiveviewSample();
    } catch (...) {
        return;
    }

    returnCode = DjiUser_GetCurrentFileDirPath(__FILE__, DJI_FILE_PATH_SIZE_MAX, curFileDirPath);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get file current path error, stat = 0x%08llX", returnCode);
    }

    // cout << "Please choose the stream demo you want to run\n\n"
    //      << "--> [0] Normal RGB image display\n"
    //      << "--> [1] Binary image display\n"
    //      << "--> [2] Faces detection demo\n"
    //      << "--> [3] Tensorflow Object detection demo\n"
    //      << endl;
    // cin >> demoIndexChar;

    // switch (demoIndexChar) {
    //     case '0':
    //         s_demoIndex = 0;
    //         break;
    //     case '1':
    //         s_demoIndex = 1;
    //         break;
    //     case '2':
    //         s_demoIndex = 2;
    //         break;
    //     case '3':
    //         s_demoIndex = 3;
    //         break;
    //     default:
    //         cout << "No demo selected";
    //         delete liveviewSample;
    //         return;
    // }

    const std::string engine_file_path{reinterpret_cast<char *>(mainName)};
    yolov8 = new YOLOv8(engine_file_path);
    yolov8->make_pipe(true);

    // ByteTrack tracker
    tracker = new BYTETracker(30, 30);
    // Detect Distance
    detdistance = new DetectDistance(fx,fy,cx,cy);//camera instr

    // cout << "Please enter the type of camera stream you want to view\n\n"
    //      << "--> [0] Fpv Camera\n"
    //      << "--> [1] Main Camera\n"
    //      << "--> [2] Vice Camera\n"
    //      << "--> [3] Top Camera\n"
    //      << endl;
    // cin >> cameraIndexChar;
    outputFolder = outputFolder + getCurrentTimeString();

    if (createDirectory(outputFolder)) {
        std::cout << "Successfully created directory: " << outputFolder << std::endl;
    }
    liveviewSample->StartMainCameraStream(&DjiUser_ShowRgbImageCallback, &mainName);
    // liveviewSample->StartMainCameraStream(&TestImageCallback, &mainName);

    // switch (cameraIndexChar) {
    //     case '0':
    //         liveviewSample->StartFpvCameraStream(&DjiUser_ShowRgbImageCallback, &fpvName);
    //         break;
    //     case '1':
    //         liveviewSample->StartMainCameraStream(&DjiUser_ShowRgbImageCallback, &mainName);
    //         break;
    //     case '2':
    //         liveviewSample->StartViceCameraStream(&DjiUser_ShowRgbImageCallback, &viceName);
    //         break;
    //     case '3':
    //         liveviewSample->StartTopCameraStream(&DjiUser_ShowRgbImageCallback, &topName);
    //         break;
    //     default:
    //         cout << "No camera selected";
    //         delete liveviewSample;
    //         return;
    // }
    // s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
    // s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();

    returnCode = osalHandler->TaskCreate("command_flying_task", DjiUser_FlightControllerCommandFlyingTask,
                                         DJI_TEST_COMMAND_FLYING_TASK_STACK_SIZE, NULL,
                                         &s_commandFlyingTaskHandle);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Create command flying task failed, errno = 0x%08llX", returnCode);
        return;
    }

    returnCode = osalHandler->TaskCreate("status_display_task", DjiUser_FlightControllerStatusDisplayTask,
                                         DJI_TEST_COMMAND_FLYING_TASK_STACK_SIZE, NULL,
                                         &s_statusDisplayTaskHandle);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Create status display task failed, errno = 0x%08llX", returnCode);
        return;
    }

    // USER_LOG_INFO("Entry \"DjiInterestPoint_RegMissionStateCallback\" before!");

    // returnCode = DjiInterestPoint_RegMissionStateCallback(DjiUser_InterestPointMissionStateCallback);
    // if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
    //     USER_LOG_ERROR("Register mission state callback failed, errno=%lld", returnCode);
    //     return;
    // }

    // USER_LOG_INFO("Entry \"DjiInterestPoint_RegMissionStateCallback\" after!");

    osalHandler->TaskSleepMs(1000);
    
        
    // Initialize the VideoWriter


    // int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Codec for .mp4
    // double fps = 30.0; // Adjust FPS as needed
    // Add timestamp to the image
    // auto now = std::chrono::system_clock::now();
    // auto in_time_t = std::chrono::system_clock::to_time_t(now);
    //std::string ss;
    //ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    // std::string outputFilePath = getCurrentTimeString() + ".mp4";
    // printf("outputFilePath: %s\n", outputFilePath.c_str());
    // USER_LOG_INFO(" - Output file path: %s", outputFilePath.c_str());
    
    // Initialize VideoWriter if not already initialized
    //if (!isVideoWriterInitialized) {
    // videoWriter.open(outputFilePath, codec, fps, pimage->size(), true);
    //    isVideoWriterInitialized = true;
    //}
    
    // std::string outputFilePath = getCurrentTimeString()
    // create ouput folder

    s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
    s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();
    // returnCode =  DjiTest_InterestPointRunSample();
    // if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
    //     USER_LOG_ERROR("Create interesting point task failed, errno = 0x%08llX", returnCode);
    //     return;
    // }
    Custom_Flight();
    // Show_Dji_POI_Error();

    // uint32_t gimbalMode = 0;
    // uint32_t rotateMode = 1;
    // // dji_f32_t pitch, roll, yaw;
    // E_DjiMountPosition gimbalMountPosition = E_DjiMountPosition(1);
    // T_DjiGimbalManagerRotation rotation;
    // // T_DjiAircraftInfoBaseInfo baseInfo;
    // // E_DjiAircraftSeries aircraftSeries;

    // DjiGimbalManager_Init();

    // rotation.rotationMode = (E_DjiGimbalRotationMode)rotateMode;
    // rotation.pitch = -90.0;
    // // rotation.pitch = 0;
    // rotation.roll = 0;
    // rotation.yaw = 0;
    // rotation.time = 2.5;
    // DjiGimbalManager_SetMode(gimbalMountPosition, (E_DjiGimbalMode)gimbalMode);
    // DjiGimbalManager_Rotate(gimbalMountPosition, rotation);

    // cout << "Please enter the 'q' or 'Q' to quit camera stream view\n"
    //      << endl;

    while (true) {
        cin >> isQuit;
        if (isQuit == 'q' || isQuit == 'Q') {
            USER_LOG_INFO(" - Quit camera stream view");
            break;
        }
    }
    USER_LOG_INFO(" - Stop camera stream");
    liveviewSample->StopMainCameraStream();
    
    // Release the VideoWriter after the loop (when you are done processing frames)
    //if (isVideoWriterInitialized) {
    // USER_LOG_INFO(" - Video saved to %s", outputFilePath.c_str());
    // videoWriter.release();
    // USER_LOG_INFO(" - VideoWriter released");

    //}

    // switch (cameraIndexChar) {
    //     case '0':
    //         liveviewSample->StopFpvCameraStream();
    //         break;
    //     case '1':
    //         liveviewSample->StopMainCameraStream();
    //         break;
    //     case '2':
    //         liveviewSample->StopViceCameraStream();
    //         break;
    //     case '3':
    //         liveviewSample->StopTopCameraStream();
    //         break;
    //     default:
    //         cout << "No camera selected";
    //         delete liveviewSample;
    //         return;
    // }

    delete liveviewSample;
    delete yolov8;
    delete tracker;
    delete detdistance;
}

/* Private functions definition-----------------------------------------------*/
static void DjiUser_ShowRgbImageCallback(CameraRGBImage img, void *userData)
{
    // chrono::time_point<chrono::system_clock, chrono::milliseconds> curtimestamp
        //  = chrono::time_point_cast<chrono::milliseconds>(chrono::system_clock::now());
    // long int curtimestamp = getTimeStamp();
    if( std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timestamp).count()
     >= 1000)
    {
            // printf("test: decode FPS[%d]\n", num);
            USER_LOG_INFO("test: decode FPS[%d]\n", num);
            // printf("test: decode FPS[%d], curtimestamp[%ld], last[%ld]\n", num, curtimestamp, timestamp);
            timestamp = std::chrono::steady_clock::now();
            num = 0;
    }
    num++;

    // string name = string(reinterpret_cast<char *>(userData));
    //const std::string engine_file_path{reinterpret_cast<char *>(userData)};
    // string path{"/home/link/Desktop/yolo_psdk/detect/100.jpg"};
    //auto yolov8 = new YOLOv8(engine_file_path);
    //yolov8->make_pipe(false);

    // cv::Mat             res, image;
    cv::Mat             image;
    cv::Size            size = cv::Size{640, 640};
    std::vector<Object> objs;

    dji_f32_t offset_x, offset_y;

    // cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    // image = cv::imread(path);
    // objs.clear();
    // yolov8->copy_from_Mat(image, size);
    // yolov8->infer();
    // yolov8->postprocess(objs);
    // yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
    // cv::imshow("result", res);
    //cv::waitKey(0);

#ifdef OPEN_CV_INSTALLED
    image = cv::Mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width * 3);
    // pimage = &image;
    
    cvtColor(image, image, COLOR_RGB2BGR);
    objs.clear();
    // image = cv::imread(path);
    yolov8->copy_from_Mat(image, size);
    // cv::imshow("show_image:", image);
    // cv::waitKey(1);
    // std::cout << "开始yolo：\n";
  
    // auto start = std::chrono::system_clock::now();
    yolov8->infer();
    // auto end = std::chrono::system_clock::now();
    yolov8->postprocess(objs);
    // track
    // void* tobjs = &objs;
    // vector<strack::Object>* pobjs = (vector<strack::Object>*)tobjs;
    std::vector<STrack> output_stracks = tracker->update(*((std::vector<strack::Object>*) &objs));

    s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
    s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();
    T_DjiVector3f gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
    gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n", gimbalAngles.x, gimbalAngles.y, gimbalAngles.z);
    USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
    if (!output_stracks.empty()) {
        USER_LOG_INFO(" - Detected objects");

    // if (!objs.empty()) {
        // offset_x = objs[0].rect.x + objs[0].rect.width / 2 - image.cols / 2;
        // offset_y = objs[0].rect.y + objs[0].rect.height / 2 - image.rows / 2;
        // x:tlwh[0],y:tlwh[1],width:tlwh[2],height:tlwh[3]
        offset_x = output_stracks[0].tlwh[0] + output_stracks[0].tlwh[2] / 2 - image.cols / 2;//image width 1920/1440
        offset_y = output_stracks[0].tlwh[1] + output_stracks[0].tlwh[3] / 2 - image.rows / 2;//image height 1080
        detdistance->UploadbboxCenter(output_stracks[0].tlwh[0] + output_stracks[0].tlwh[2] / 2 ,output_stracks[0].tlwh[1] + output_stracks[0].tlwh[3] / 2);

        if (0 < offset_x) {
            // DjiFlightController_CancelLanding();
            s_flyingCommand.y = s_flyingSpeed;
        }else if (0 > offset_x) {
            // DjiFlightController_CancelLanding();
            s_flyingCommand.y = -s_flyingSpeed;
        }else {
            s_flyingCommand.y = 0;
        }

        if (0 < offset_y) {
            // DjiFlightController_CancelLanding();
            s_flyingCommand.x = -s_flyingSpeed;
        }else if (0 > offset_y) {
            // DjiFlightController_CancelLanding();
            s_flyingCommand.x = s_flyingSpeed;
        }else {
            s_flyingCommand.x = 0;
        }

        // USER_LOG_INFO(boundmin)
        if (boundmin <= offset_x && boundmax >= offset_x && boundmin <= offset_y && boundmax >= offset_y) {
            DjiFlightController_StartLanding();
            
            USER_LOG_INFO(" - Start landing\r\n Due to the object  offset_x: %f, offset_y: %f", offset_x, offset_y);

            DjiGimbalManager_Reset(E_DjiMountPosition(1), DJI_GIMBAL_RESET_MODE_PITCH_AND_YAW);
        }
        
        // printf("x: %f y: %f\n", objs[0].rect.x + objs[0].rect.width / 2,objs[0].rect.y + objs[0].rect.height / 2);
        cv::Scalar s = tracker->get_color(output_stracks[0].track_id);
        cv::putText(image, cv::format("track_id:%d", output_stracks[0].track_id), cv::Point(output_stracks[0].tlwh[0], output_stracks[0].tlwh[1] - 5), 
                    0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Rect(output_stracks[0].tlwh[0], output_stracks[0].tlwh[1], output_stracks[0].tlwh[2], output_stracks[0].tlwh[3]), s, 2);
        // file name format: "year-month-day hour:minute:second"
        
    }
    // std::cout << "image.size:" <<image.size();
    // Write the frame to the video file
    // videoWriter.write(image);

    // std::string outputFilePath = getCurrentTimeString() + ".jpg";
    // std::string outputFilePath = "/home/nvidia/Desktop/yolo_psdk/output/" + getCurrentTimeString() + num + ".jpg";
    cv::imwrite(outputFolder+"/"+std::to_string(img_count) +".jpg", image);
    img_count++;
#endif
}

static void TestImageCallback(CameraRGBImage img, void *userData){
    if( std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timestamp).count()
     >= 1000)
    {
            // printf("test: decode FPS[%d]\n", num);
            USER_LOG_INFO("test: decode FPS[%d]\n", num);
            // printf("test: decode FPS[%d], curtimestamp[%ld], last[%ld]\n", num, curtimestamp, timestamp);
            timestamp = std::chrono::steady_clock::now();
            num = 0;
    }
    num++;
    cv::Mat image = cv::Mat(img.height, img.width, CV_8UC3, img.rawData.data(), img.width * 3);
    cvtColor(image, image, COLOR_RGB2BGR);
    cv::imshow("result", image);
    cv::waitKey(1);
}

static T_DjiReturnCode DjiUser_GetCurrentFileDirPath(const char *filePath, uint32_t pathBufferSize, char *dirPath)
{
    uint32_t i = strlen(filePath) - 1;
    uint32_t dirPathLen;

    while (filePath[i] != '/') {
        i--;
    }

    dirPathLen = i + 1;

    if (dirPathLen + 1 > pathBufferSize) {
        return DJI_ERROR_SYSTEM_MODULE_CODE_INVALID_PARAMETER;
    }

    memcpy(dirPath, filePath, dirPathLen);
    dirPath[dirPathLen] = 0;

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static void *DjiUser_FlightControllerCommandFlyingTask(void *arg)
{
    T_DjiReturnCode returnCode;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiFlightControllerRidInfo ridInfo = {0};
    T_DjiFlightControllerGeneralInfo generalInfo = {0};

    ridInfo.latitude = 22.542812;
    ridInfo.longitude = 113.958902;
    ridInfo.altitude = 0;

    returnCode = DjiFlightController_Init(ridInfo);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Init flight controller failed, errno = 0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Init data subscription module failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    /*! subscribe fc data */
    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_QUATERNION,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_50_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic flight status failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_GIMBAL_ANGLES,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_50_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic gimbal angles failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_GPS_POSITION,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_10_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic gps failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_HEIGHT_FUSION,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_10_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic height fusion failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_ALTITUDE_FUSED,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_50_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic altitude fused failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_ALTITUDE_OF_HOMEPOINT,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_5_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic altitude of homepoint failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_HOME_POINT_INFO,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_10_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic info of homepoint failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_POSITION_VO,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_50_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic positionVO failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    returnCode = DjiFcSubscription_SubscribeTopic(DJI_FC_SUBSCRIPTION_TOPIC_CONTROL_DEVICE,
                                                  DJI_DATA_SUBSCRIPTION_TOPIC_5_HZ,
                                                  NULL);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Subscribe topic control device failed, error code:0x%08llX", returnCode);
        return NULL;
    }

    osalHandler->TaskSleepMs(1000);

    // returnCode = DjiUser_FlightControlUpdateConfig();
    // if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
    //     USER_LOG_ERROR("Update config failed, error code:0x%08llX", returnCode);
    // }

    returnCode = DjiFlightController_GetGeneralInfo(&generalInfo);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get general info failed, error code:0x%08llX", returnCode);
    }
    USER_LOG_INFO("Get aircraft serial number is: %s", generalInfo.serialNum);

    returnCode = DjiFlightController_RegJoystickCtrlAuthorityEventCallback(
        DjiUser_FlightCtrlJoystickCtrlAuthSwitchEventCb);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS && returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_NONSUPPORT) {
        USER_LOG_ERROR("Register joystick control authority event callback failed, errno = 0x%08llX", returnCode);
        return NULL;
    }

    isCommandFlyingTaskStart = true;

    while (true) {
        s_inputFlag++;
        if (s_inputFlag > 25) {
            s_flyingCommand.x = 0;
            s_flyingCommand.y = 0;
            s_flyingCommand.z = 0;
            s_flyingCommand.yaw = 0;
            s_inputFlag = 0;
        }

        DjiUser_FlightControllerVelocityAndYawRateCtrl(s_flyingCommand);

        osalHandler->TaskSleepMs(1000 / DJI_TEST_COMMAND_FLYING_CTRL_FREQ);
    }
}

static void *DjiUser_FlightControllerStatusDisplayTask(void *arg)
{
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    while (1) {
        if (isCommandFlyingTaskStart == false) {
            continue;
        }
#ifdef OPEN_CV_INSTALLED
        DjiUser_ShowFlightStatusByOpenCV();
#endif
        osalHandler->TaskSleepMs(1000 / DJI_TEST_COMMAND_FLYING_CTRL_FREQ);
    }
}

static void DjiUser_ShowFlightStatusByOpenCV(void)
{
#ifdef OPEN_CV_INSTALLED
    E_DjiFlightControllerGoHomeAltitude goHomeAltitude = 0;
    T_DjiVector3f aircraftAngles = {0};
    T_DjiFcSubscriptionAltitudeOfHomePoint altitudeOfHomePoint = {0};
    E_DjiFlightControllerRtkPositionEnableStatus rtkPositionEnableStatus;
    E_DjiFlightControllerRCLostAction rcLostAction = DJI_FLIGHT_CONTROLLER_RC_LOST_ACTION_HOVER;
    E_DjiFlightControllerObstacleAvoidanceEnableStatus downwardsVisEnable;
    E_DjiFlightControllerObstacleAvoidanceEnableStatus upwardsVisEnable;
    E_DjiFlightControllerObstacleAvoidanceEnableStatus horizontalVisEnable;
//    E_DjiFlightControllerObstacleAvoidanceEnableStatus upwardsRadarEnable;
//    E_DjiFlightControllerObstacleAvoidanceEnableStatus horizontalRadarEnable;
    T_DjiFcSubscriptionControlDevice controlDevice;
    T_DjiFcSubscriptionPositionVO positionVo;
    T_DjiAircraftInfoBaseInfo aircraftInfoBaseInfo;
    T_DjiReturnCode returnCode;

    returnCode = DjiAircraftInfo_GetBaseInfo(&aircraftInfoBaseInfo);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("get aircraft base info error");
    }

    Mat img(480, 1000, CV_8UC1, cv::Scalar(0));

    // Get latest flight status
    if (aircraftInfoBaseInfo.aircraftSeries != DJI_AIRCRAFT_SERIES_M300) {
        DjiFlightController_GetRCLostAction(&rcLostAction);
    }
    DjiFlightController_GetGoHomeAltitude(&s_goHomeAltitude);
    DjiFlightController_GetRtkPositionEnableStatus(&rtkPositionEnableStatus);
    DjiFlightController_GetDownwardsVisualObstacleAvoidanceEnableStatus(&downwardsVisEnable);
//    DjiFlightController_GetUpwardsRadarObstacleAvoidanceEnableStatus(&upwardsRadarEnable);
    DjiFlightController_GetUpwardsVisualObstacleAvoidanceEnableStatus(&upwardsVisEnable);
//    DjiFlightController_GetHorizontalRadarObstacleAvoidanceEnableStatus(&horizontalRadarEnable);
    DjiFlightController_GetHorizontalVisualObstacleAvoidanceEnableStatus(&horizontalVisEnable);

    controlDevice = DjiUser_FlightControlGetValueOfControlDevice();
    aircraftAngles = DjiUser_FlightControlGetValueOfQuaternion();
    s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
    altitudeOfHomePoint = DjiUser_FlightControlGetValueOfRelativeHeight();
    positionVo = DjiUser_FlightControlGetValueOfPositionVo();

    if (s_statusDisplayTaskCnt++ % 20 == 0) {
        singleBatteryInfo1 = DjiUser_FlightControlGetValueOfBattery1();
        singleBatteryInfo2 = DjiUser_FlightControlGetValueOfBattery2();
    }

    // Display latest flight status
    cv::putText(img, "Status: ", cv::Point(30, 20), FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 0, 0));

    cv::putText(img, "Roll: " + cv::format("%.4f", aircraftAngles.y), cv::Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 0, 0));
    cv::putText(img, "Pitch: " + cv::format("%.4f", aircraftAngles.x), cv::Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 0, 0));
    cv::putText(img, "Yaw: " + cv::format("%.4f", aircraftAngles.z), cv::Point(50, 110), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 0, 0));
    cv::putText(img, "WorldX: " + cv::format("%.4f", positionVo.x), cv::Point(50, 140), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 0, 0));
    cv::putText(img, "WorldY: " + cv::format("%.4f", positionVo.y), cv::Point(50, 170), FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(200, 0, 0));
    cv::putText(img, "WorldZ: " + cv::format("%.4f", altitudeOfHomePoint), cv::Point(50, 200), FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "Latitude: " + cv::format("%.4f", (dji_f64_t) s_gpsPosition.y / 10000000), cv::Point(50, 230),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "Longitude: " + cv::format("%.4f", (dji_f64_t) s_gpsPosition.x / 10000000), cv::Point(50, 260),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "Battery1: " + cv::format("%d%%", singleBatteryInfo1.batteryCapacityPercent), cv::Point(50, 290),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "Battery2: " + cv::format("%d%%", singleBatteryInfo2.batteryCapacityPercent), cv::Point(50, 320),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));

    cv::putText(img, "Config: ", cv::Point(300, 20), FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 0, 0));
    cv::putText(img, "-> RcLostAction(Sync APP): " + cv::format("%d  (0-hover 1-landing 2-gohome)", rcLostAction),
                cv::Point(320, 50),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> GoHomeAltitude(Sync APP): " + cv::format("%d", s_goHomeAltitude), cv::Point(320, 80),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> RTK-Enable(Sync APP): " + cv::format("%d", rtkPositionEnableStatus), cv::Point(320, 110),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> HomePointLatitude: " + cv::format("%.4f", s_homeLocation.latitude), cv::Point(320, 140),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> HomePointLongitude: " + cv::format("%.4f", s_homeLocation.longitude), cv::Point(320, 170),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> FlyingSpeed: " + cv::format("%.2f", s_flyingSpeed), cv::Point(320, 200),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> downwardsVisEnable(Sync APP): " + cv::format("%d", downwardsVisEnable), cv::Point(320, 230),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> upwardsVisEnable(Sync APP): " + cv::format("%d", upwardsVisEnable), cv::Point(320, 260),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> horizontalVisEnable(Sync APP): " + cv::format("%d", horizontalVisEnable), cv::Point(320, 290),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));
    cv::putText(img, "-> ControlDevice: " + cv::format("%d", controlDevice.deviceStatus), cv::Point(320, 320),
                FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 0, 0));

    cv::putText(img,
                "[Q]-Up    [W]-Front  [E]-Down   [R]-TakeOff  [T]-CancelLanding  [Y]-CancelGoHome  [I]-ArrestFly  [O]-CancelArrestFly  [P]-EmgStopMotor",
                cv::Point(30, 400), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(128, 0, 0));
    cv::putText(img,
                "[A]-Left   [S]-Near   [D]-Right   [F]-ForceLand   [G]-Landing   [H]-GoHome  [J]-UpdateConfig  [K]-Brake  [L]-CancelBrakeI",
                cv::Point(30, 430), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(128, 0, 0));
    cv::putText(img,
                "[Z]-Yaw-  [X]-RefreshHomePoint   [C]-Yaw+  [V]-ConfirmLanding   [B]-TurnOn  [N]-TurnOff  [M]-ObtainCtrlAuth",
                cv::Point(30, 460), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(128, 0, 0));

    cv::imshow("Payload SDK Command Flying Data Observation Window", img);
    cv::waitKey(1);
#endif
}

static void DjiUser_FlightControllerVelocityAndYawRateCtrl(T_DjiFlightControllerJoystickCommand command)
{
    T_DjiFlightControllerJoystickMode joystickMode = {
        DJI_FLIGHT_CONTROLLER_HORIZONTAL_VELOCITY_CONTROL_MODE,
        DJI_FLIGHT_CONTROLLER_VERTICAL_VELOCITY_CONTROL_MODE,
        DJI_FLIGHT_CONTROLLER_YAW_ANGLE_RATE_CONTROL_MODE,
        DJI_FLIGHT_CONTROLLER_HORIZONTAL_BODY_COORDINATE,
        DJI_FLIGHT_CONTROLLER_STABLE_CONTROL_MODE_ENABLE,
    };
    T_DjiReturnCode returnCode;

    DjiFlightController_SetJoystickMode(joystickMode);

    USER_LOG_DEBUG("Joystick command: %.2f %.2f %.2f", command.x, command.y, command.z);
    returnCode = DjiFlightController_ExecuteJoystickAction(command);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Execute joystick command failed, errno = 0x%08llX", returnCode);
        return;
    }
}

static T_DjiReturnCode
DjiUser_FlightCtrlJoystickCtrlAuthSwitchEventCb(T_DjiFlightControllerJoystickCtrlAuthorityEventInfo eventData)
{
    switch (eventData.joystickCtrlAuthoritySwitchEvent) {
        case DJI_FLIGHT_CONTROLLER_MSDK_GET_JOYSTICK_CTRL_AUTH_EVENT: {
            if (eventData.curJoystickCtrlAuthority == DJI_FLIGHT_CONTROLLER_JOYSTICK_CTRL_AUTHORITY_MSDK) {
                USER_LOG_INFO("[Event] Msdk request to obtain joystick ctrl authority\r\n");
            } else {
                USER_LOG_INFO("[Event] Msdk request to release joystick ctrl authority\r\n");
            }
            break;
        }
        case DJI_FLIGHT_CONTROLLER_INTERNAL_GET_JOYSTICK_CTRL_AUTH_EVENT: {
            if (eventData.curJoystickCtrlAuthority == DJI_FLIGHT_CONTROLLER_JOYSTICK_CTRL_AUTHORITY_INTERNAL) {
                USER_LOG_INFO("[Event] Internal request to obtain joystick ctrl authority\r\n");
            } else {
                USER_LOG_INFO("[Event] Internal request to release joystick ctrl authority\r\n");
            }
            break;
        }
        case DJI_FLIGHT_CONTROLLER_OSDK_GET_JOYSTICK_CTRL_AUTH_EVENT: {
            if (eventData.curJoystickCtrlAuthority == DJI_FLIGHT_CONTROLLER_JOYSTICK_CTRL_AUTHORITY_OSDK) {
                USER_LOG_INFO("[Event] Request to obtain joystick ctrl authority\r\n");
            } else {
                USER_LOG_INFO("[Event] Request to release joystick ctrl authority\r\n");
            }
            break;
        }
        case DJI_FLIGHT_CONTROLLER_RC_LOST_GET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to rc lost\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_RC_NOT_P_MODE_RESET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc for rc is not in P mode\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_RC_SWITCH_MODE_GET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to rc switching mode\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_RC_PAUSE_GET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to rc pausing\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_RC_REQUEST_GO_HOME_GET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to rc request for return\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_LOW_BATTERY_GO_HOME_RESET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc for low battery return\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_LOW_BATTERY_LANDING_RESET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc for low battery land\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_OSDK_LOST_GET_JOYSTICK_CTRL_AUTH_EVENT:
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to sdk lost\r\n");
            break;
        case DJI_FLIGHT_CONTROLLER_NERA_FLIGHT_BOUNDARY_RESET_JOYSTICK_CTRL_AUTH_EVENT :
            USER_LOG_INFO("[Event] Current joystick ctrl authority is reset to rc due to near boundary\r\n");
            break;
        default:
            USER_LOG_INFO("[Event] Unknown joystick ctrl authority event\r\n");
    }

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static T_DjiFcSubscriptionGpsPosition DjiUser_FlightControlGetValueOfGpsPosition(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionGpsPosition gpsPosition;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_GPS_POSITION,
                                                      (uint8_t *) &gpsPosition,
                                                      sizeof(T_DjiFcSubscriptionGpsPosition),
                                                      &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic gps position error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return gpsPosition;
}

static T_DjiFcSubscriptionAltitudeOfHomePoint DjiUser_FlightControlGetValueOfRelativeHeight(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionAltitudeOfHomePoint altitudeOfHomePoint;
    T_DjiFcSubscriptionAltitudeFused altitudeFused;
    
    // djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_ALTITUDE_FUSED,
    //                                             (uint8_t *) &altitudeOfHomePoint,
    //                                             sizeof(T_DjiFcSubscriptionAltitudeOfHomePoint),
    //                                             &timestamp);

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_ALTITUDE_OF_HOMEPOINT,
                                                    (uint8_t *) &altitudeOfHomePoint,
                                                    sizeof(T_DjiFcSubscriptionAltitudeOfHomePoint),
                                                    &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic altitude of homepoint error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                        timestamp.microsecond);
    }


    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_ALTITUDE_FUSED,
                                                    (uint8_t *) &altitudeFused,
                                                    sizeof(T_DjiFcSubscriptionAltitudeFused),
                                                    &timestamp);

    // djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_HEIGHT_FUSION,
    //                                                   (uint8_t *) &altitudeOfHomePoint,
    //                                                   sizeof(T_DjiFcSubscriptionAltitudeOfHomePoint),
    //                                                   &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic altitude fusion error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return (altitudeFused - altitudeOfHomePoint);
}

static T_DjiFcSubscriptionHomePointInfo DjiUser_FlightControlGetValueOfHomepointInfo(void){
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionHomePointInfo homePointInfo;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_HOME_POINT_INFO,
                                                    (uint8_t *) &homePointInfo,
                                                    sizeof(T_DjiFcSubscriptionHomePointInfo),
                                                    &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic homepoint info error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                        timestamp.microsecond);
    }

    return homePointInfo;
}

static T_DjiFcSubscriptionPositionVO DjiUser_FlightControlGetValueOfPositionVo(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionPositionVO positionVo;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_POSITION_VO,
                                                      (uint8_t *) &positionVo,
                                                      sizeof(T_DjiFcSubscriptionPositionVO),
                                                      &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic quaternion error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return positionVo;
}

static T_DjiFcSubscriptionControlDevice DjiUser_FlightControlGetValueOfControlDevice(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionControlDevice controlDevice;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_CONTROL_DEVICE,
                                                      (uint8_t *) &controlDevice,
                                                      sizeof(T_DjiFcSubscriptionControlDevice),
                                                      &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic quaternion error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return controlDevice;
}

static T_DjiVector3f DjiUser_FlightControlGetValueOfQuaternion(void)
{
    T_DjiReturnCode djiStat;
    T_DjiFcSubscriptionQuaternion quaternion = {0};
    T_DjiDataTimestamp quaternionTimestamp = {0};
    dji_f64_t pitch, yaw, roll;
    T_DjiVector3f vector3F;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_QUATERNION,
                                                      (uint8_t *) &quaternion,
                                                      sizeof(T_DjiFcSubscriptionQuaternion),
                                                      &quaternionTimestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic quaternion error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", quaternionTimestamp.millisecond,
                       quaternionTimestamp.microsecond);
        USER_LOG_DEBUG("Quaternion: %f %f %f %f.", quaternion.q0, quaternion.q1, quaternion.q2, quaternion.q3);
    }

    pitch = (dji_f64_t) asinf(-2 * quaternion.q1 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q2) * 57.3;
    roll = (dji_f64_t) atan2f(2 * quaternion.q2 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q1,
                              -2 * quaternion.q1 * quaternion.q1 - 2 * quaternion.q2 * quaternion.q2 + 1) * 57.3;
    yaw = (dji_f64_t) atan2f(2 * quaternion.q1 * quaternion.q2 + 2 * quaternion.q0 * quaternion.q3,
                             -2 * quaternion.q2 * quaternion.q2 - 2 * quaternion.q3 * quaternion.q3 + 1) *
          57.3;

    vector3F.x = pitch;
    vector3F.y = roll;
    vector3F.z = yaw;

    return vector3F;
}

static T_DjiVector3f DjiUser_FlightControlGetValueOfGimbalAngles(void)
{
    T_DjiReturnCode djiStat;
    // T_DjiFcSubscriptionQuaternion quaternion = {0};
    T_DjiFcSubscriptionGimbalAngles gimbalAngles = {0};
    T_DjiDataTimestamp eulerAnglesTimestamp = {0};
    // dji_f64_t pitch, yaw, roll;
    T_DjiVector3f vector3F;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_GIMBAL_ANGLES,
                                                      (uint8_t *) &gimbalAngles,
                                                      sizeof(T_DjiFcSubscriptionGimbalAngles),
                                                      &eulerAnglesTimestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic gimbalangles error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", eulerAnglesTimestamp.millisecond,
                       eulerAnglesTimestamp.microsecond);
        // USER_LOG_DEBUG("Quaternion: %f %f %f %f.", quaternion.q0, quaternion.q1, quaternion.q2, quaternion.q3);
    }

    // pitch = (dji_f64_t) asinf(-2 * quaternion.q1 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q2) * 57.3;
    // roll = (dji_f64_t) atan2f(2 * quaternion.q2 * quaternion.q3 + 2 * quaternion.q0 * quaternion.q1,
    //                           -2 * quaternion.q1 * quaternion.q1 - 2 * quaternion.q2 * quaternion.q2 + 1) * 57.3;
    // yaw = (dji_f64_t) atan2f(2 * quaternion.q1 * quaternion.q2 + 2 * quaternion.q0 * quaternion.q3,
    //                          -2 * quaternion.q2 * quaternion.q2 - 2 * quaternion.q3 * quaternion.q3 + 1) *
    //       57.3;

    vector3F.x = gimbalAngles.x;//pitch
    vector3F.y = gimbalAngles.y;//roll
    vector3F.z = gimbalAngles.z;//yaw

    return vector3F;
}

static T_DjiFcSubscriptionSingleBatteryInfo DjiUser_FlightControlGetValueOfBattery1(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionSingleBatteryInfo singleBatteryInfo;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_BATTERY_SINGLE_INFO_INDEX1,
                                                      (uint8_t *) &singleBatteryInfo,
                                                      sizeof(T_DjiFcSubscriptionSingleBatteryInfo),
                                                      &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic battery1 error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return singleBatteryInfo;
}

static T_DjiFcSubscriptionSingleBatteryInfo DjiUser_FlightControlGetValueOfBattery2(void)
{
    T_DjiReturnCode djiStat;
    T_DjiDataTimestamp timestamp = {0};
    T_DjiFcSubscriptionSingleBatteryInfo singleBatteryInfo;

    djiStat = DjiFcSubscription_GetLatestValueOfTopic(DJI_FC_SUBSCRIPTION_TOPIC_BATTERY_SINGLE_INFO_INDEX2,
                                                      (uint8_t *) &singleBatteryInfo,
                                                      sizeof(T_DjiFcSubscriptionSingleBatteryInfo),
                                                      &timestamp);

    if (djiStat != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Get value of topic battery2 error, error code: 0x%08X", djiStat);
    } else {
        USER_LOG_DEBUG("Timestamp: millisecond %u microsecond %u.", timestamp.millisecond,
                       timestamp.microsecond);
    }

    return singleBatteryInfo;
}

static void Custom_Flight(void){
    T_DjiReturnCode returnCode;
    // uint32_t gimbalMode = 0;
    // uint32_t rotateMode = 1;
    // // dji_f32_t pitch, roll, yaw;
    // E_DjiMountPosition gimbalMountPosition = E_DjiMountPosition(1);
    T_DjiGimbalManagerRotation rotation;
    T_DjiVector3f gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    T_DjiFcSubscriptionHomePointInfo homePointInfo;
    T_DjiInterestPointSettings interestPointSettings = {0};
    interestPointSettings.latitude = 22.542812;
    interestPointSettings.longitude = 113.958902;
    // // T_DjiAircraftInfoBaseInfo baseInfo;
    // // E_DjiAircraftSeries aircraftSeries;

    DjiGimbalManager_Init();
    returnCode = DjiInterestPoint_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest init failed, errno=%lld", returnCode);
        return;
    }
    returnCode = DjiInterestPoint_RegMissionStateCallback(DjiUser_InterestPointMissionStateCallback);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Register mission state callback failed, errno=%lld", returnCode);
        return;
    }

    // rotation.rotationMode = (E_DjiGimbalRotationMode)rotateMode;
    // rotation.pitch = -90.0;
    // rotation.roll = 0;
    // rotation.yaw = 0;
    // rotation.time = 2.5;
    // DjiGimbalManager_SetMode(gimbalMountPosition, (E_DjiGimbalMode)gimbalMode);
    // DjiGimbalManager_Rotate(gimbalMountPosition, rotation);

    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiFcSubscriptionAltitudeOfHomePoint altitudeOfHomePoint = {0};
    // T_DjiFcSubscriptionPositionVO positionVo;

    DjiFlightController_ObtainJoystickCtrlAuthority();
    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
    // gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    USER_LOG_INFO("Current homepoint longitude(经度):%frad latitude(纬度):%frad\n", homePointInfo.longitude,homePointInfo.latitude);
    USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n", gimbalAngles.x, gimbalAngles.y, gimbalAngles.z);
    USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
    DjiFlightController_StartTakeoff();
    USER_LOG_INFO(" - Take off\r\n");
    USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
    gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n", gimbalAngles.x, gimbalAngles.y, gimbalAngles.z);
    USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
    rotation.rotationMode = (E_DjiGimbalRotationMode)1;
    rotation.pitch = -90.0;
    rotation.roll = 0;
    rotation.yaw = 0;
    rotation.time = 2.5;
    //云台模式决定了云台跟随无人机运动时的转动方式：
    // 自由模式：当无人机的姿态改变时，云台将不会转动。
    // FPV 模式：当无人机的姿态发生改变时，云台会转动航向轴与横滚轴，确保负载设备当前的视场角不会发生改变。
    // YAW 跟随模式：在该模式下，云台的航向轴会跟随无人机的航向轴转动。
    // DjiGimbalManager_SetMode(gimbalMountPosition, (E_DjiGimbalMode)gimbalMode);
    DjiGimbalManager_Rotate(E_DjiMountPosition(1), rotation);
    // osalHandler->TaskSleepMs(1000*10);
    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    USER_LOG_INFO("Current homepoint longitude(经度):%frad latitude(纬度):%frad\n", homePointInfo.longitude,homePointInfo.latitude);
    USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n",gimbalAngles.x ,gimbalAngles.y ,gimbalAngles.z);
    USER_LOG_INFO("entry customflight while");

    while (true) {
        osalHandler->TaskSleepMs(1);
    
        s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
        altitudeOfHomePoint = DjiUser_FlightControlGetValueOfRelativeHeight();
        s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();

        // printf("altitudeOfHomePoint:%.4f", altitudeOfHomePoint);

        if (2 > altitudeOfHomePoint) {
            // s_flyingCommand.z = s_flyingSpeed / 10;
            s_flyingCommand.z = 0.5;
            s_inputFlag = 0;
            std::ostringstream logStream;
            logStream << " - UP: " << std::fixed << std::setprecision(4) << altitudeOfHomePoint << "\r\n";
            USER_LOG_INFO("%s\r\n", logStream.str().c_str());
            USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
            gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
            USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n", gimbalAngles.x, gimbalAngles.y, gimbalAngles.z);
            USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
            // printf("%s\r\n", logStream.str().c_str());
        }
        else if (2 > s_positionVo.x){
            s_flyingCommand.z = 0;
            s_flyingCommand.x = 0.5;
            s_inputFlag = 0;
            //if (!detdistance->BBoxIsEmpty()) {
            //goto InterestPoint;
            //}
            std::ostringstream logStream;
            logStream << " - Front: " << std::fixed << std::setprecision(4) << s_positionVo.x  << "\r\n";
            USER_LOG_INFO("%s\r\n", logStream.str().c_str());
            USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
            gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
            USER_LOG_INFO("Current gimbal angles pitch:%f roll:%f yaw:%f\n", gimbalAngles.x, gimbalAngles.y, gimbalAngles.z);
            USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
            // printf("%s\r\n", logStream.str().c_str());
        }
        else {
            s_flyingCommand.x = 0;
            // goto Dji_Rot;
            goto InterestPoint;
            return;
            // DjiGimbalManager_Reset(gimbalMountPosition, DJI_GIMBAL_RESET_MODE_PITCH_AND_YAW);
        }

    }
InterestPoint:

    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    // s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
    altitudeOfHomePoint = DjiUser_FlightControlGetValueOfRelativeHeight();
    s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();
    gimbalAngles = DjiUser_FlightControlGetValueOfGimbalAngles();
    detdistance->UploadCameraHeight(altitudeOfHomePoint);
    detdistance->UploadOriginGPS(homePointInfo.latitude,homePointInfo.longitude);
    detdistance->UploadGimbalAngles(*((F_Vector3f*)&gimbalAngles));
    detdistance->CalculateDistance();
    // D_Vector3f wgs84 = detdistance->outputWgs84();
    //detdistance->HoverPoint(interestPointSettings.latitude,interestPointSettings.longitude);
    USER_LOG_INFO("Current target longitude(经度):%f° latitude(纬度):%f°\n", interestPointSettings.longitude ,interestPointSettings.latitude);

    DjiInterestPoint_SetSpeed(1.0f);

    returnCode = DjiInterestPoint_Start(interestPointSettings);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest start failed, errno=%lld", returnCode);
        return;
    }

    for (int i = 0; i < 20; ++i) {
        USER_LOG_INFO("Interest point mission running %ds.", i);
        osalHandler->TaskSleepMs(1000);
    }

    returnCode = DjiInterestPoint_Stop();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest stop failed, errno=%lld", returnCode);
        return;
    }
    return;

Dji_Rot:
    // printf("this is start of DjiGimbalManager_Rotate\n");
    rotation.rotationMode = (E_DjiGimbalRotationMode)1;
    rotation.pitch = -90.0;
    rotation.roll = 0;
    rotation.yaw = 0;
    rotation.time = 2.5;
    // DjiGimbalManager_SetMode(gimbalMountPosition, (E_DjiGimbalMode)gimbalMode);
    DjiGimbalManager_Rotate(E_DjiMountPosition(1), rotation);
    // printf("this is end of DjiGimbalManager_Rotate\n");
    return;

}

static T_DjiReturnCode DjiUser_InterestPointMissionStateCallback(T_DjiInterestPointMissionState missionState)
{
    USER_LOG_INFO("Interest point state: %d, radius: %.2f m, speed: %.2f m/s", missionState.state, missionState.radius,
                  missionState.curSpeed);

    return DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS;
}

static void Show_Dji_POI_Error(void){
    T_DjiReturnCode returnCode;
    T_DjiFcSubscriptionHomePointInfo homePointInfo;
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    T_DjiInterestPointSettings interestPointSettings = {0};
    interestPointSettings.latitude = 22.542812;
    interestPointSettings.longitude = 113.958902;
    returnCode = DjiInterestPoint_Init();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest init failed, errno=%lld", returnCode);
        return;
    }
    returnCode = DjiInterestPoint_RegMissionStateCallback(DjiUser_InterestPointMissionStateCallback);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Register mission state callback failed, errno=%lld", returnCode);
        return;
    }
    T_DjiFcSubscriptionAltitudeOfHomePoint altitudeOfHomePoint = {0};
    DjiFlightController_ObtainJoystickCtrlAuthority();
    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
    USER_LOG_INFO("Current homepoint longitude(经度):%frad latitude(纬度):%frad\n", homePointInfo.longitude,homePointInfo.latitude);
    USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
    DjiFlightController_StartTakeoff();
    USER_LOG_INFO(" - Take off\r\n");
    USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
    USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    USER_LOG_INFO("Current homepoint longitude(经度):%frad latitude(纬度):%frad\n", homePointInfo.longitude,homePointInfo.latitude);
    USER_LOG_INFO("entry customflight while");
    while (true) {
        osalHandler->TaskSleepMs(1);
        s_gpsPosition = DjiUser_FlightControlGetValueOfGpsPosition();
        altitudeOfHomePoint = DjiUser_FlightControlGetValueOfRelativeHeight();
        s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();
        if (2 > altitudeOfHomePoint) {
            s_flyingCommand.z = 0.5;
            s_inputFlag = 0;
            std::ostringstream logStream;
            logStream << " - UP: " << std::fixed << std::setprecision(4) << altitudeOfHomePoint << "\r\n";
            USER_LOG_INFO("%s\r\n", logStream.str().c_str());
            USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
            USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
        }
        else if (2 > s_positionVo.x){
            s_flyingCommand.z = 0;
            s_flyingCommand.x = 0.5;
            s_inputFlag = 0;
            std::ostringstream logStream;
            logStream << " - Front: " << std::fixed << std::setprecision(4) << s_positionVo.x  << "\r\n";
            USER_LOG_INFO("%s\r\n", logStream.str().c_str());
            USER_LOG_INFO("Current position x(北):%f y(东):%f z(地):%f\n", s_positionVo.x, s_positionVo.y, s_positionVo.z);
            USER_LOG_INFO("Current gps longitude:%f latitude:%f altitude:%f\n", dji_f64_t(s_gpsPosition.x)*1e-7, dji_f64_t(s_gpsPosition.y)*1e-7, dji_f64_t(s_gpsPosition.z)*1e-3);
        }
        else {
            s_flyingCommand.x = 0;
            goto InterestPoint;
            return;
        }

    }
InterestPoint:
    homePointInfo = DjiUser_FlightControlGetValueOfHomepointInfo();
    altitudeOfHomePoint = DjiUser_FlightControlGetValueOfRelativeHeight();
    s_positionVo = DjiUser_FlightControlGetValueOfPositionVo();
    USER_LOG_INFO("Current target longitude(经度):%f° latitude(纬度):%f°\n", interestPointSettings.longitude ,interestPointSettings.latitude);
    DjiInterestPoint_SetSpeed(1.0f);
    returnCode = DjiInterestPoint_Start(interestPointSettings);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest start failed, errno=%lld", returnCode);
        return;
    }
    for (int i = 0; i < 20; ++i) {
        USER_LOG_INFO("Interest point mission running %ds.", i);
        osalHandler->TaskSleepMs(1000);
    }
    returnCode = DjiInterestPoint_Stop();
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS) {
        USER_LOG_ERROR("Point interest stop failed, errno=%lld", returnCode);
        return;
    }
    return;
}

/****************** (C) COPYRIGHT DJI Innovations *****END OF FILE****/
