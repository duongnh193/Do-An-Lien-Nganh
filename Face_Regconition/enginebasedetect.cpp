#include "enginebasedetect.h"

EngineBase::EngineBase() {
    modelShape = cv::Size(320,320);
    inputNodeDims= {1,3,320, 320};
}

void EngineBase::InitEngine()
{
    LOG_C("from base class");
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Vivas");
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOption.SetIntraOpNumThreads(numThread);
    sessionOption.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    engine = new Ort::Session(env, modelPath.c_str(),sessionOption);
    LOG_C("init done");
}
