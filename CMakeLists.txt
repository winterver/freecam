cmake_minimum_required(VERSION 3.5)
project(freecam)

if(WIN32)
    set(GLFW_HEADER C:/Users/I/vimfiles/.library)
    set(GLFW_LIBRARY C:/Users/I/vimfiles/.library)
    set(VULKAN_SDK D:/Application/VulkanSDK/1.3.283.0)
    if(NOT EXISTS ${GLFW_HEADER})
        message(FATAL_ERROR "Please set library paths")
    elseif(NOT EXISTS ${GLFW_LIBRARY})
        message(FATAL_ERROR "Please set library paths")
    elseif(NOT EXISTS ${VULKAN_SDK})
        message(FATAL_ERROR "Please set library paths")
    endif()
    include_directories(${GLFW_HEADER} ${VULKAN_SDK}/Include)
    link_directories(${GLFW_LIBRARY} ${VULKAN_SDK}/Lib)
endif()

add_executable(freecam
               src/main.cpp
               src/header_only_impl.cpp)
target_link_libraries(freecam glfw3)

if(MSVC)
    target_compile_options(freecam PRIVATE /MD)
endif()

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
                      -Dout=${PROJECT_SOURCE_DIR}/src/embeded_shaders.hpp
                      -P ${PROJECT_SOURCE_DIR}/EmbedShaders.cmake)

add_dependencies(freecam embed_shaders)
add_dependencies(embed_shaders compile_shaders)
