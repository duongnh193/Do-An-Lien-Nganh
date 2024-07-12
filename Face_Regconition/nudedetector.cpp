#include "nudedetector.h"

NudeDetector::NudeDetector() {}

void NudeDetector::InitEngine()
{
    runOption = Ort::RunOptions{nullptr};
    EngineBase::InitEngine();
    Ort::AllocatorWithDefaultOptions allocator;
    uint inputCount = engine->GetInputCount();
    for (size_t i = 0; i < inputCount; i++) {
        Ort::AllocatedStringPtr inputNodeName = engine->GetInputNameAllocated(i, allocator);
        char *temp_buf = new char[50];
        strcpy(temp_buf, inputNodeName.get());
        inputNames.push_back(temp_buf);
    }
    uint outputCount = engine->GetOutputCount();
    for (size_t i = 0; i < outputCount; i++) {
        Ort::AllocatedStringPtr outNames = engine->GetOutputNameAllocated(i, allocator);
        char *temp_buf = new char[50];
        strcpy(temp_buf, outNames.get());
        outputNames.push_back(temp_buf);
    }
}

std::vector<float> NudeDetector::InferenceEngine(cv::Mat &image)
{
    cv::Mat preprocessImg = PreprocessImage(image);
    cv::Mat blobImg = cv::dnn::blobFromImage(preprocessImg,NORM);
    float *imgData = (float*) blobImg.data;
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<float*>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,OrtMemTypeCPU), imgData, 3 * modelShape.height * modelShape.width,
        inputNodeDims.data(),inputNodeDims.size());
    auto sessionResult = engine->Run(runOption, inputNames.data(), &inputTensor, inputNames.size(),outputNames.data(), outputNames.size());
    Ort::TypeInfo typeInfo = sessionResult.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto outputData = sessionResult.front().GetTensorMutableData<typename std::remove_pointer<float*>::type>();
    LOG_C(outputNodeDims.size());
    cv::Mat output = cv::Mat(outputNodeDims[1], outputNodeDims[2], CV_32F, outputData);
    output = output.t();
    LOG_C("width: " << output.cols << " height: " << output.rows);
    PosprocessImage(image,output);
}

cv::Mat NudeDetector::PreprocessImage(cv::Mat &image)
{
    int maxSize = image.cols > image.rows ? image.cols : image.rows;
    cv::Mat result(maxSize, maxSize, image.type());
    image.copyTo(result(cv::Rect(0,0,image.cols, image.rows)));
    cv::cvtColor(result,result,cv::COLOR_BGR2RGB);
    cv::resize(result,result, modelShape);
    return result;
}

std::vector<NudeDetector::NudeInfo> NudeDetector::PosprocessImage(cv::Mat &image, cv::Mat &sessionResult)
{
    LOG_C(sessionResult.cols << " " << sessionResult.rows);
    std::vector<NudeInfo> results;
    float *data = (float*) sessionResult.data;
    int stride = sessionResult.cols;
    float factor =  image.cols > image.rows ? (float)image.cols/modelShape.width : (float)image.rows/modelShape.height;
    int rows = sessionResult.rows;
    for(uint i=0; i<rows ; i++){
        float *classesScores = data + 4;
        cv::Mat scores(1, stride-4, CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if(maxClassScore >= objectThreshold){
            NudeInfo info;
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            //            LOG_INFO << "x y w h: " << x << " " << y << " "<< w << " " << h;
            info.bbox.x = float((x - 0.5 * w) * factor);
            info.bbox.y = float((y - 0.5 * h) * factor);

            info.bbox.width = float(w * factor);
            info.bbox.height = float(h * factor);
            //            LOG_INFO << "facetor: " << factor << " " << image.size;
            //correct
            info.bbox.x = MAX(info.bbox.x, 0);
            info.bbox.y = MAX(info.bbox.y,0);
            info.bbox.width = MIN(info.bbox.width, image.cols - info.bbox.x);
            info.bbox.height = MIN(info.bbox.height, image.rows - info.bbox.y);
            //            LOG_INFO << "x y w h: " << vi.bbox.x << " " << vi.bbox.y << " "<< vi.bbox.width << " " << vi.bbox.height;
            //            LOG_INFO <<"\n\n";
            info.confident = maxClassScore;
            info.classId = class_id.x;
            info.className = classes[info.classId];

            results.push_back(info);
        }
        data += stride;
    }

    results = NMS(results);
    LOG_C("detect size: " << results.size());

    //drawing
    cv::Mat drawing = image.clone();
    for(auto info : results){
        if(info.className == "FEMALE_GENITALIA_EXPOSED" || info.className == "FEMALE_BREAST_COVERED" || info.className == "FACE_FEMALE"){
            cv::rectangle(drawing, info.bbox,cv::Scalar(255,0,0),2);
        }
    }
    cv::imshow("img", drawing);
    cv::waitKey();
    return results;
}

std::vector<NudeDetector::NudeInfo> NudeDetector::NMS(std::vector<NudeInfo> bboxes)
{
    std::vector<NudeInfo> result;
    bool overlap =false;
    cv::Rect2f intersec;
    cv::Rect2f uni;
    float IOU;
    for(const auto &bbox : bboxes){
        overlap =false;
        for(auto &bbox2: result){
            intersec = bbox.bbox & bbox2.bbox;
            uni = bbox.bbox | bbox2.bbox;
            IOU = intersec.area()/uni.area();

            if(IOU >= IouThreshold){
                overlap = true;
                if (bbox.confident > bbox2.confident){
                    bbox2 = bbox;
                }
                break;
            }
        }
        if(!overlap){
            result.push_back(bbox);
        }
    }
    return result;
}
