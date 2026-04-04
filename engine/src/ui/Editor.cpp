#include "ui/Editor.h"
#include <imgui.h>

void Editor::Render() {
    // Enable Dockspace
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    // Control Panel
    ImGui::Begin("PINN Controls");
    ImGui::Text("Fluid Dynamics Parameters");
    ImGui::SliderFloat("Inflow Velocity", &m_InflowVelocity, 0.0f, 10.0f);
    ImGui::SliderFloat("Kinematic Viscosity", &m_Viscosity, 0.001f, 0.1f, "%.4f");
    ImGui::Separator();
    
    ImGuiIO& io = ImGui::GetIO();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();

    // Viewport Placeholder
    ImGui::Begin("3D Viewport");
    ImGui::Text("OpenGL Render Target will go here...");
    ImGui::End();
}