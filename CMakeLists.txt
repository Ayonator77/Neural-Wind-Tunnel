cmake_minimum_required(VERSION 3.20)
project(NeuralWindTunnel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------
# vcpkg Dependencies
# ---------------------------------------------------------
find_package(SDL2   CONFIG REQUIRED)
find_package(glad   CONFIG REQUIRED)
find_package(glm    CONFIG REQUIRED)
find_package(GTest  CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# ---------------------------------------------------------
# ImGui — docking branch via FetchContent
# ---------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.4-docking
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_docking STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui_docking PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_compile_definitions(imgui_docking PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD=1)
target_link_libraries(imgui_docking PUBLIC SDL2::SDL2 glad::glad)

# ---------------------------------------------------------
# Engine Core Library
# ---------------------------------------------------------
add_library(engine_core
    # We will add your core engine files here as we build them out
    # e.g., src/core/Application.cpp, src/inference/PinnModel.cpp
    engine/src/main.cpp # Temporary: Just to satisfy CMake until we split it
)

target_include_directories(engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(engine_core PUBLIC
    SDL2::SDL2main
    SDL2::SDL2
    glad::glad
    glm::glm
    imgui_docking
    spdlog::spdlog
    onnxruntime::onnxruntime
)

# ---------------------------------------------------------
# Main Executable
# ---------------------------------------------------------
# Once we split main.cpp from engine_core, add it here.
add_executable(WindTunnelApp engine/src/main.cpp) 
target_link_libraries(WindTunnelApp PRIVATE engine_core)

set_target_properties(WindTunnelApp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

# Copy runtime DLLs next to the exe (needed for ONNX on Windows)
add_custom_command(TARGET WindTunnelApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:WindTunnelApp>
        $<TARGET_FILE_DIR:WindTunnelApp>
    COMMAND_EXPAND_LISTS
)

# Create and copy assets directory
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/assets")
add_custom_command(TARGET WindTunnelApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:WindTunnelApp>/assets"
)

# ---------------------------------------------------------
# Custom Run Target
# ---------------------------------------------------------
add_custom_target(run
    COMMAND $<TARGET_FILE:WindTunnelApp>
    DEPENDS WindTunnelApp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ---------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------
enable_testing()
# add_executable(engine_tests tests/test_main.cpp)
# target_link_libraries(engine_tests PRIVATE engine_core GTest::gtest) 
# include(GoogleTest)
# gtest_discover_tests(engine_tests)