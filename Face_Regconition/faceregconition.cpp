#include "faceregconition.h"

FaceRegconition::FaceRegconition() {}

void FaceRegconition::InitEngine2()
{
    runOption = Ort::RunOptions{nullptr};
    EngineBase2::InitEngine2();
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
    for (size_t i = 0; i < inputNames.size(); i++) {
        LOG_C("Value at index" << i << ":" << inputNames[i]);
    }
    for (size_t i = 0; i < outputNames.size(); i++) {
        LOG_C("Value at index " << i << ":" << outputNames[i]);
    }

}


std::vector<float> FaceRegconition::InferenceEngine2(cv::Mat &image)
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
    // LOG_C(outputNodeDims.size());
    cv::Mat output = cv::Mat(outputNodeDims[0], outputNodeDims[1], CV_32F, outputData);
    output = output.t();
    // LOG_C("width: " << output.cols << " height: " << output.rows);
    std::vector<float> vecto;
    if (output.isContinuous()) {
        vecto.assign(output.ptr<float>(), output.ptr<float>() + output.total());
    } else {
        for (int i = 0; i < output.rows; ++i) {
            vecto.insert(vecto.end(), output.ptr<float>(i), output.ptr<float>(i) + output.cols);
        }
    }
    return vecto;
}

float FaceRegconition::CousineDistance(std::vector<float> v1, std::vector<float> v2) {
    if (v1.size() != v2.size()) {
        LOG_C("not same size");
    }

    float dot_product = 0.0;
    float norm_v1 = 0.0;
    float norm_v2 = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }

    return dot_product / (std::sqrt(norm_v1) * std::sqrt(norm_v2));
}

cv::Mat FaceRegconition::PreprocessImage(cv::Mat &image)
{
    int maxSize = image.cols > image.rows ? image.cols : image.rows;
    cv::Mat result(maxSize, maxSize, image.type());
    image.copyTo(result(cv::Rect(0,0,image.cols, image.rows)));
    cv::cvtColor(result,result,cv::COLOR_BGR2RGB);
    cv::resize(result,result, modelShape);
    return result;
}
