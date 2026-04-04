#pragma once
#include <SDL2/SDL.h>
#include <memory>
#include "../ui/Editor.h"

class Application {
public:
    Application();
    ~Application();

    void Run();

private:
    void Init();
    void Shutdown();
    void HandleEvents();

    SDL_Window* m_Window;
    SDL_GLContext m_GLContext;
    bool m_IsRunning;

    std::unique_ptr<Editor> m_Editor;
};