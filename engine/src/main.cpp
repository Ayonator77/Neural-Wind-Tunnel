#include <iostream>
#include <SDL2/SDL.h>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PINN_Test");
    std::cout << "SDL2 and ONNX Runtime initialized successfully!" << std::endl;

    SDL_Quit();
    return 0;
}