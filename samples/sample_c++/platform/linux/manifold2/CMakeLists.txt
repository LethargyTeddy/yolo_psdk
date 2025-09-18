cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(dji_sdk_demo_linux_cxx LANGUAGES C CXX CUDA)

# Disable in-source builds to prevent source tree corruption.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.")
endif ()

set(CMAKE_C_FLAGS "-pthread -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_EXE_LINKER_FLAGS "-pthread")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
add_definitions(-DSYSTEM_ARCH_LINUX -D_GNU_SOURCE)

set(LIBRARY_PATH psdk_lib/lib/aarch64-linux-gnu-gcc)

install(FILES ${LIBRARY_PATH}/libpayloadsdk.a
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        )

        message("\n${CMAKE_INSTALL_PREFIX}")

install(DIRECTORY psdk_lib/include
        DESTINATION "${CMAKE_INSTALL_PREFIX}"
        )

if (MEMORY_LEAK_CHECK_ON MATCHES TRUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lasan")
endif ()

if (BUILD_TEST_CASES_ON MATCHES TRUE)
    set(COMMON_CXX_FLAGS "-std=c++11 -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -Wno-deprecated-declarations")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -lgcov")
endif ()

include_directories(../../../module_sample)
include_directories(../../../../sample_c/module_sample)
include_directories(../common/osal)
include_directories(../manifold2/application)

file(GLOB_RECURSE MODULE_SAMPLE_SRC
        ../../../module_sample/liveview/*.c*
        ../../../../sample_c/module_sample/*.c
        )
file(GLOB_RECURSE MODULE_COMMON_SRC ../common/osal/*.c*)
file(GLOB_RECURSE MODULE_HAL_SRC hal/*.c*)
file(GLOB_RECURSE MODULE_APP_SRC application/*.c*)
file(GLOB_RECURSE bytetrack_SRC ../../../../../bytetrack/src/*.cpp)

execute_process(COMMAND uname -m
        OUTPUT_VARIABLE DEVICE_SYSTEM_ID)
string(STRIP ${DEVICE_SYSTEM_ID} DEVICE_SYSTEM_ID)
if (DEVICE_SYSTEM_ID MATCHES x86_64)
    set(TOOLCHAIN_NAME x86_64-linux-gnu-gcc)
    add_definitions(-DPLATFORM_ARCH_x86_64=1)
elseif (DEVICE_SYSTEM_ID MATCHES aarch64)
    set(TOOLCHAIN_NAME aarch64-linux-gnu-gcc)
    add_definitions(-DPLATFORM_ARCH_aarch64=1)
else ()
    message(FATAL_ERROR "FATAL: Please confirm your platform.")
endif ()

message("\ncmak当前资源文件：${CMAKE_CURRENT_SOURCE_DIR}\n")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../common/3rdparty)

message("\ncmake当前列表文件：${CMAKE_CURRENT_LIST_DIR}\n")
link_directories(${CMAKE_CURRENT_LIST_DIR}/psdk_lib/lib/${TOOLCHAIN_NAME})
link_libraries(${CMAKE_CURRENT_LIST_DIR}/psdk_lib/lib/${TOOLCHAIN_NAME}/libpayloadsdk.a -lstdc++)

# link_directories(${CMAKE_CURRENT_LIST_DIR}/../../../../../psdk_lib/lib/${TOOLCHAIN_NAME})
# link_libraries(${CMAKE_CURRENT_LIST_DIR}/../../../../../psdk_lib/lib/${TOOLCHAIN_NAME}/libpayloadsdk.a -lstdc++)

message("\nMODULE_COMMON_SRC：${MODULE_COMMON_SRC}")
add_executable(${PROJECT_NAME}
        ${MODULE_APP_SRC}
        ${MODULE_SAMPLE_SRC}
        # message("\nMODULE_COMMON_SRC：${MODULE_COMMON_SRC}")
        ${MODULE_COMMON_SRC}
        ${MODULE_HAL_SRC}
        ${bytetrack_SRC}
        # main.cpp
        ../../../../../detect/include/yolov8.hpp
        ../../../../../detect/include/common.hpp
        )

# Try to see if user has OpenCV installed
# if yes, default callback will display the image

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /usr/lib/aarch64-linux-gnu)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
    message("\n${PROJECT_NAME}...")
    message(STATUS "Found OpenCV installed in the system, will use it to display image in AdvancedSensing APIs")
    message(STATUS " - Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS " - Libraries: ${OpenCV_LIBRARIES}")
    add_definitions(-DOPEN_CV_INSTALLED)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else ()
    message(STATUS "Did not find OpenCV in the system, image data is inside RecvContainer as raw data")
endif ()

find_package(FFMPEG REQUIRED)
if (FFMPEG_FOUND)
    message(STATUS "Found FFMPEG installed in the system")
    message(STATUS " - Includes: ${FFMPEG_INCLUDE_DIR}")
    message(STATUS " - Libraries: ${FFMPEG_LIBRARIES}")

    EXECUTE_PROCESS(COMMAND ffmpeg -version
            OUTPUT_VARIABLE ffmpeg_version_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    string(REGEX MATCH "version.*Copyright" ffmpeg_version_line ${ffmpeg_version_output})
    string(REGEX MATCH " .* " ffmpeg_version ${ffmpeg_version_line})
    string(REGEX MATCH "^ 5.*$" ffmpeg_major_version ${ffmpeg_version})

    if (HEAD${ffmpeg_major_version} STREQUAL "HEAD")
        message(STATUS " - Version: ${ffmpeg_version}")
    else ()
        message(FATAL_ERROR " - Not support FFMPEG version: ${ffmpeg_major_version}, please install 4.x.x instead.")
    endif ()

    target_link_libraries(${PROJECT_NAME} ${FFMPEG_LIBRARIES})
    include_directories(${FFMPEG_INCLUDE_DIR})
    add_definitions(-DFFMPEG_INSTALLED)
else ()
    message(STATUS "Cannot Find FFMPEG")
endif (FFMPEG_FOUND)
include_directories(${CMAKE_CURRENT_LIST_DIR}/psdk_lib/include)

find_package(OPUS REQUIRED)
if (OPUS_FOUND)
    message(STATUS "Found OPUS installed in the system")
    message(STATUS " - Includes: ${OPUS_INCLUDE_DIR}")
    message(STATUS " - Libraries: ${OPUS_LIBRARY}")

#    add_definitions(-DOPUS_INSTALLED)
#    target_link_libraries(${PROJECT_NAME} /usr/local/lib/libopus.a)
else ()
    message(STATUS "Cannot Find OPUS")
endif (OPUS_FOUND)

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ../../../../../detect/include
        ../../../../../bytetrack/include
        /usr/include/eigen3
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})

link_directories(${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} nvinfer nvinfer_plugin ${CUDA_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin ${CUDA_LIBRARIES} ${OpenCV_LIBS})

find_package(LIBUSB REQUIRED)
if (LIBUSB_FOUND)
    message(STATUS "Found LIBUSB installed in the system")
    message(STATUS " - Includes: ${LIBUSB_INCLUDE_DIR}")
    message(STATUS " - Libraries: ${LIBUSB_LIBRARY}")

    add_definitions(-DLIBUSB_INSTALLED)
    target_link_libraries(${PROJECT_NAME} usb-1.0)
else ()
    message(STATUS "Cannot Find LIBUSB")
endif (LIBUSB_FOUND)

if (NOT EXECUTABLE_OUTPUT_PATH)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
endif ()

target_link_libraries(${PROJECT_NAME} m)
