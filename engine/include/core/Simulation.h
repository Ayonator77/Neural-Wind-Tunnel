#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

struct FluidState{
    float u, v, w; // Velocity components
    float p;      // Pressure
};

class Simulation {
public:
    // Pass the path to the .onnx file when creating the simulation
    Simulation(const std::string& modelPath);
    ~Simulation() = default;

    //Queries the nerual network for the fluid state at specific coordinates and time
    FluidState SamplePINN(float x, float y, float z, float t);

private:
    Ort::Env m_Env;
    Ort::Session m_Session{nullptr};
    Ort::MemoryInfo m_MemoryInfo;
    
    // ONNX I/O names required by the API
    std::vector<const char*> m_InputNames;
    std::vector<const char*> m_OutputNames;
};