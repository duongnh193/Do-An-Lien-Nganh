#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <QtNetwork/QNetworkAccessManager>
#include <onnxruntime_cxx_api.h>
#include <enginebasedetect.h>
#include <nudedetector.h>
#include "facedetect.h"
#include "faceregconition.h"
#include "readfile.h"
#include"LoadDataFace.h"
#include <QTextStream>
#include <QFile>



int main(int argc, char *argv[])
{

    // QCoreApplication app(argc, argv);
    // NudeDetector nudeDetect;
    // nudeDetect.modelPath = "/home/duongnh/Data/ModelTrained/NudeNet.onnx";
    // nudeDetect.modelShape = cv::Size(320,320);
    // nudeDetect.InitEngine();
    // cv::Mat img = cv::imread("/home/duongnh/Downloads/img test/bikini.jpeg");
    // nudeDetect.InferenceEngine(img);


    FaceDetect faceDetect;
    faceDetect.modelPath = "/home/duongnh/SourceCode/yolov8_detection1.onnx";
    faceDetect.modelShape = cv::Size(640,640);
    faceDetect.InitEngine();

    FaceRegconition faceRegconition;
    faceRegconition.modelPath = "/home/duongnh/Data/ModelTrained/face_recognition_sface_2021dec.onnx";
    faceRegconition.modelShape = cv::Size(112,112);
    faceRegconition.InitEngine2();

    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceID = 0; // 0 = open default camera
    int apiID = cv::CAP_ANY; // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    ReadFile readFile;
    LoadDataFace loadData;
    auto userData = loadData.LoadUserData();
    //--- GRAB AND WRITE LOOP
    std::cout << "Start grabbing" << std::endl
         << "Press any key to terminate" << std::endl;
    while(true)
    {
        // // wait for a new frame from camera and store it into 'frame'
        // cap.read(frame);
        // // check if we succeeded
        // if (frame.empty()) {
        //     std::cerr << "ERROR! blank frame grabbed\n";
        //     break;
        // }
        // auto faces = faceDetect.InferenceEngine(frame);
        // if (faces.empty()) {
        //     cv::putText(frame, "No face detected", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        // } else {
        //     for (cv::Rect &rect : faces){
        //         cv::rectangle(frame, rect, cv::Scalar(0,0,255),2);
        //     }
        //     std::vector<float> dataFace = faceRegconition.InferenceEngine2(frame);
        //     // std::string faceName = "Name: " + std::to_string(faces.size());

        //     std::string numFaces = std::to_string(faces.size()) + " face detected";
        //     readFile.ReadData(numFaces,dataFace);
        //     cv::putText(frame, numFaces, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // }
        // imshow("Frame", frame);
        // if (cv::waitKey(1) >= 0){
        //     break;
        // }

        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        auto faces = faceDetect.InferenceEngine(frame);
        if (faces.empty()) {
            cv::putText(frame, "No face detected", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        } else {
            for (cv::Rect &rect : faces) {
                cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2);

                cv::Mat faceROI = frame(rect);
                std::vector<float> dataFace = faceRegconition.InferenceEngine2(faceROI);

                bool found = false;
                std::string detectedName;
                float saveSimilarity = 0.0;
                for (auto &user : userData) {
                    float similarity = faceRegconition.CousineDistance(user.second, dataFace);
                    if(similarity > saveSimilarity){
                        saveSimilarity = similarity;
                    }
                    if (similarity > 0.9) {
                        detectedName = user.first;
                        found = true;
                        break;
                    }
                }

                if (!found && saveSimilarity < 0.9) {
                    std::cout << "New face detected. Please enter your name: ";
                    std::cin >> detectedName;
                    loadData.SaveUserData(detectedName, dataFace);
                    userData[detectedName] = dataFace;
                }
                cv::putText(frame, detectedName, cv::Point(rect.x + 20, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
            std::string numFaces = std::to_string(faces.size()) + " face detected";
            cv::putText(frame, numFaces, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
        imshow("Frame", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    cap.release();
    // ReadFile readFile;
    // readFile.ReadData();
    // return app.exec();
}


