cmake_minimum_required(VERSION 3.5)
project(freecam)

if(WIN32)
    set(VULKAN_HOME D:/Application/VulkanSDK/1.3.283.0)
    include_directories(lib ${VULKAN_HOME}/Include)
    link_directories(lib ${VULKAN_HOME}/Lib)
endif()

add_executable(freecam
               src/main.cpp
               src/volk.c)

if(WIN32)
    target_compile_options(freecam PRIVATE /MD)
endif()
target_link_libraries(freecam glfw3)

include(FetchContent)
FetchContent_Declare(cmake-spirv
                     GIT_REPOSITORY https://github.com/liliolett/cmake-spirv.git
                     GIT_TAG        origin/v1)
FetchContent_MakeAvailable(cmake-spirv)
list(APPEND CMAKE_MODULE_PATH ${cmake-spirv_SOURCE_DIR}/include)
include(AddSpirvModules)

add_spirv_modules(compile_shaders
                  BINARY_DIR shaders
                  SOURCE_DIR src/shaders
                  SOURCES    shader.vert shader.frag
                  OPTIONS    -O)

add_custom_target(embed_shaders
                  COMMAND ${CMAKE_COMMAND}
                      -Ddir=${CMAKE_CURRENT_BINARY_DIR}/shaders
                      -Dout=${PROJECT_SOURCE_DIR}/src/embeded_shaders.h
                      -P ${PROJECT_SOURCE_DIR}/EmbedShaders.cmake)

add_dependencies(freecam embed_shaders)
add_dependencies(embed_shaders compile_shaders)
