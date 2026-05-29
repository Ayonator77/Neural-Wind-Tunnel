#include "core/Simulation.h"
#include <iostream>
#include <stdexcept>

Simulation::Simulation(const std::string& modelPath)
:m_Env(ORT_LOGGING_LEVEL_WARNING, "NeuralWindTunnel"),
m_MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    try{
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        // ONNX Runtime expects a wide string (wstring) for file paths on Windows
        std::wstring w_modelPath = std::wstring(modelPath.begin(), modelPath.end());
        m_Session = Ort::Session(m_Env, w_modelPath.c_str(), session_options);

        std::cout << "Successfully loaded PINN Model: " << modelPath << std::endl;

        // Set the input and output node names exactly as exported from PyTorch
        m_InputNames = { "input_coords" };
        m_OutputNames = { "fluid_preds" };
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load ONNX model.");
    }

}