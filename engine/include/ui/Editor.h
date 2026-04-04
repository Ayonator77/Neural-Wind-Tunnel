#pragma once

class Editor {
public:
    Editor() = default;
    ~Editor() = default;

    void Render();

private:
    float m_InflowVelocity = 1.0f;
    float m_Viscosity = 0.01f;
};