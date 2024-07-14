#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <array>
#include <unordered_map>

#include <Volk/volk.h>
#include <vma/vk_mem_alloc.h>

#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/hash.hpp>

#include "tiny_obj_loader.hpp"
#include "embeded_shaders.hpp"
#include "stb_image.h"

template<class T>
uint32_t sizeof_container(T container)
{
    return uint32_t(sizeof(container[0]) * container.size());
}

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

struct UniformBlock
{
    glm::mat4 MVP;
    glm::mat4 uModel;
    glm::vec3 viewPos;
};

namespace _ {
    using namespace glm;

    // edited from https://www.shadertoy.com/view/XlGcRh
    vec2 grad(vec2 p2, int seed)
    {
        vec3 p3 = vec3(p2, seed);
        p3 = fract(p3 * vec3(0.1031f, 0.1030f, 0.0973f));
        vec3 yzx = vec3(p3.y, p3.z, p3.x);
        p3 += dot(p3, yzx+33.33f);
        vec2 xx = vec2(p3.x, p3.x);
        vec2 yz = vec2(p3.y, p3.z);
        vec2 zy = vec2(p3.z, p3.y);
        return fract((xx+yz)*zy) * 2.0f - 1.0f;
    }

    // from https://godotshaders.com/snippet/2d-noise/
    // return value range [0, 1]
    float noise(vec2 uv, int seed = 1337) {
        vec2 uv_index = floor(uv);
        vec2 uv_fract = fract(uv);

        vec2 blur = smoothstep(0.0f, 1.0f, uv_fract);

        return mix( mix( dot( grad( uv_index + vec2(0.0f,0.0f), seed ), uv_fract - vec2(0.0f,0.0f) ),
                         dot( grad( uv_index + vec2(1.0f,0.0f), seed ), uv_fract - vec2(1.0f,0.0f) ), blur.x),
                    mix( dot( grad( uv_index + vec2(0.0f,1.0f), seed ), uv_fract - vec2(0.0f,1.0f) ),
                         dot( grad( uv_index + vec2(1.0f,1.0f), seed ), uv_fract - vec2(1.0f,1.0f) ), blur.x), blur.y) + 0.5f;
    }

    // edited from https://godotshaders.com/snippet/fractal-brownian-motion-fbm/
    // note to call fbm() like fbm(pos/64.0f), or the result may not be what you expect
    float fbm(vec2 uv, int seed = 1337, int octaves = 6, float amplitude = 0.5f, float gain = 0.5f, float frequency = 1.0f, float lacunarity = 2.0f) {
        float value = 0.0f;
        for(int i = 0; i < octaves; i++) {
            value += amplitude * noise(uv * frequency, seed);
            frequency *= lacunarity;
            amplitude *= gain;
        }
        return value;
    }
}

using _::noise;
using _::fbm;

class Camera
{
public:
    float fov = 60.0f;
    float vertical = 0.0f;
    float horizontal = glm::pi<float>();
    float speed = 3.0f;
    float mouseSpeed = 0.002f;
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 5.0f);

    glm::mat4 update(GLFWwindow* window)
    {
        static double lastTime = glfwGetTime();
        double currentTime = glfwGetTime();
        float deltaTime = float(currentTime - lastTime);
        lastTime = currentTime;

        static double lastX = 0, lastY = 0;
        if (lastX == 0 && lastY == 0) {
            glfwGetCursorPos(window, &lastX, &lastY);
        }
        double currentX, currentY;
        glfwGetCursorPos(window, &currentX, &currentY);
        double dx = currentX - lastX;
        double dy = currentY - lastY;
        lastX = currentX;
        lastY = currentY;

        horizontal -= float(mouseSpeed * dx);
        vertical   -= float(mouseSpeed * dy);
        horizontal = glm::mod(horizontal, glm::two_pi<float>());
        vertical = glm::clamp(vertical, -1.57f, 1.57f);

        glm::vec3 direction(
            cos(vertical) * sin(horizontal),
            sin(vertical),
            cos(vertical) * cos(horizontal)
        );

        glm::vec3 forward(
            sin(horizontal),
            0,
            cos(horizontal)
        );

        glm::vec3 right(
            sin(horizontal - glm::half_pi<float>()),
            0,
            cos(horizontal - glm::half_pi<float>())
        );

        glm::vec3 up = glm::vec3(0, 1, 0);

        glm::vec3 velocity{};
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
            velocity += forward;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
            velocity -= forward;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
            velocity += right;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
            velocity -= right;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS){
            velocity += up;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
            velocity -= up;
        }
        if (glm::length(velocity) > 0) {
            position += glm::normalize(velocity) * speed * deltaTime;
        }

        int fbWdith, fbHeight;
        glfwGetFramebufferSize(window, &fbWdith, &fbHeight);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)fbWdith/fbHeight, 0.1f, 1000.0f);
        glm::mat4 view       = glm::lookAt(position, position + direction, glm::vec3(0, 1, 0));

        return projection * view;
    }
};

class Application
{
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
private:
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    GLFWwindow* window;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice;
    int graphicsFamilyIndex = -1;
    int presentFamilyIndex = -1;

    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VmaAllocator allocator;

    VkSwapchainKHR swapchain;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkImage depthImage;
    VkImageView depthImageView;
    VmaAllocation depthImageAlloc;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    //VkPipeline blockPipeline;

    std::vector<VkFramebuffer> swapchainFramebuffers;

    VkBuffer vertexBuffer;
    VmaAllocation vertexAlloc;
    VkBuffer indexBuffer;
    VmaAllocation indexAlloc;
    int indexCount;

    VkImage albedoMap;
    VmaAllocation albedoAlloc;
    VkImageView albedoView;
    VkSampler albedoSampler;

    VkImage normalMap;
    VmaAllocation normalAlloc;
    VkImageView normalView;
    VkSampler normalSampler;

    VkImage metallicMap;
    VmaAllocation metallicAlloc;
    VkImageView metallicView;
    VkSampler metallicSampler;

    VkImage roughnessMap;
    VmaAllocation roughnessAlloc;
    VkImageView roughnessView;
    VkSampler roughnessSampler;

    //VkBuffer blockBuffer;
    //VmaAllocation blockAlloc;
    //int blockCount;
    
    VkBuffer uniformBuffer;
    VmaAllocation uniformAlloc;
    VmaAllocationInfo uniformInfo;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    Camera camera;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;

public:
    Application(GLFWwindow* window)
        : window(window)
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Freecam";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Vulkan API";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        // get Vulkan extensions required by glfw
        uint32_t glfwExtensionCount;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        // add to list our required extensions
        std::vector<const char*> requiredExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        requiredExtensions.push_back("VK_EXT_debug_utils");

        // get extensions supported by Vulkan
        uint32_t extensionCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

        std::string errlog;
        // check if required extensions exist
        for (const auto& extensionName : requiredExtensions) {
            auto find_ext = [=](const VkExtensionProperties& extension) {
                return !strcmp(extensionName, extension.extensionName);
            };
            auto it = std::find_if(availableExtensions.begin(), availableExtensions.end(), find_ext);
            if (it == availableExtensions.end()) {
                errlog += std::string("\t") + extensionName + "\n";
            }
        }

        if (!errlog.empty()) {
            std::string prefix = "Requested extension not supported:\n";
            throw std::runtime_error(prefix + errlog);
        }

        const std::vector<const char*> requiredValidationLayers = {
            "VK_LAYER_KHRONOS_validation",
            //"VK_LAYER_LUNARG_monitor", // display FPS in title bar
        };
        // get validation layers supported by Vulkan
        // they can be used to debug during development
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        errlog.clear();
        // check if required validation layers exist
        for (const auto& layerName : requiredValidationLayers) {
            auto find_layer = [=](const VkLayerProperties& layer) {
                return !strcmp(layerName, layer.layerName);
            };
            auto it = std::find_if(availableLayers.begin(), availableLayers.end(), find_layer);
            if (it == availableLayers.end()) {
                errlog += std::string("\t") + layerName + "\n";
            }
        }

        if (!errlog.empty()) {
            std::string prefix = "Requested validation layer not supported:\n";
            throw std::runtime_error(prefix + errlog);
        }

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        instanceInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        instanceInfo.ppEnabledExtensionNames = requiredExtensions.data();
        instanceInfo.enabledLayerCount = (uint32_t)requiredValidationLayers.size();
        instanceInfo.ppEnabledLayerNames = requiredValidationLayers.data();

        if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Vulkan instance");
        }

        // load extension functions
        volkLoadInstance(instance);

        // setup debug call back
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                              | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                              | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = [](
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) -> VkBool32
        {
            std::cerr << pCallbackData->pMessage << std::endl;
            return VK_FALSE;
        };

        if (vkCreateDebugUtilsMessengerEXT(instance, &debugInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
           throw std::runtime_error("Failed to setup debug messenger");
        }

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }

        findGPU();
        createDevice();
        createSwapchain();
        createDepthImage();
        createPipelines();
        createFramebuffers();
        createCommandPool();
        createBuffers();
        createDescriptors();
    }

    void findGPU()
    {
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        if (devices.size() == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        // power of GPU: discrete > integrated > other
        std::vector<VkPhysicalDevice> discreteGPUs;
        std::vector<VkPhysicalDevice> integratedGPUs;
        std::vector<VkPhysicalDevice> otherGPUs;

        for (const auto& device : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(device, &props);

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                discreteGPUs.push_back(device);
            }
            else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                integratedGPUs.push_back(device);
            }
            else {
                otherGPUs.push_back(device);
            }
        }

        std::vector<VkPhysicalDevice> mostPowerfulGPUs;
        if (!discreteGPUs.empty()) {
            mostPowerfulGPUs = discreteGPUs;
        }
        else if (!integratedGPUs.empty()) {
            mostPowerfulGPUs = integratedGPUs;
        }
        else {
            mostPowerfulGPUs = otherGPUs;
        }

        auto find_max_heap = [](const VkPhysicalDevice& lhs, const VkPhysicalDevice& rhs) {
            VkPhysicalDeviceMemoryProperties lhsProps;
            VkPhysicalDeviceMemoryProperties rhsProps;
            vkGetPhysicalDeviceMemoryProperties(lhs, &lhsProps);
            vkGetPhysicalDeviceMemoryProperties(rhs, &rhsProps);

            uint32_t lhsHeapCount = lhsProps.memoryHeapCount;
            uint32_t rhsHeapCount = rhsProps.memoryHeapCount;
            VkMemoryHeap* lhsHeaps = lhsProps.memoryHeaps;
            VkMemoryHeap* rhsHeaps = rhsProps.memoryHeaps;

            auto find_local_heap = [](const VkMemoryHeap& heap) {
                return heap.flags & VkMemoryHeapFlagBits::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
            };
            VkMemoryHeap* lhsLocalHeap = std::find_if(lhsHeaps, lhsHeaps + lhsHeapCount, find_local_heap);
            VkMemoryHeap* rhsLocalHeap = std::find_if(rhsHeaps, rhsHeaps + rhsHeapCount, find_local_heap);

            return lhsLocalHeap->size < rhsLocalHeap->size;
        };
        physicalDevice = *std::max_element(mostPowerfulGPUs.begin(), mostPowerfulGPUs.end(), find_max_heap);

        graphicsFamilyIndex = -1;
        presentFamilyIndex = -1;
        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        int index = 0;
        for (const auto& family : queueFamilies) {
            if (graphicsFamilyIndex < 0 && family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsFamilyIndex = index;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, index, surface, &presentSupport);
            if (presentFamilyIndex < 0 && index != graphicsFamilyIndex && presentSupport) {
                presentFamilyIndex = index;
            }
            index++;
        }

        if (graphicsFamilyIndex < 0 || presentFamilyIndex < 0) {
            throw std::runtime_error("Requested queue families not supported");
        }
    }

    void createDevice()
    {
        float priorities[] = { 1.0f };
        std::array<VkDeviceQueueCreateInfo, 2> queueCreateInfos{};
        queueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[0].queueFamilyIndex = graphicsFamilyIndex;
        queueCreateInfos[0].pQueuePriorities = priorities;
        queueCreateInfos[0].queueCount = 1;
        queueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfos[1].queueFamilyIndex = presentFamilyIndex;
        queueCreateInfos[1].pQueuePriorities = priorities;
        queueCreateInfos[1].queueCount = 1;

        const std::vector<const char*> requiredExtensions = {
            "VK_KHR_swapchain",
            "VK_KHR_maintenance1",
        };

        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        std::string errlog;
        for (const auto& extensionName : requiredExtensions) {
            auto find_ext = [=](const VkExtensionProperties& extension) {
                return !strcmp(extensionName, extension.extensionName);
            };
            auto it = std::find_if(availableExtensions.begin(), availableExtensions.end(), find_ext);
            if (it == availableExtensions.end()) {
                errlog += std::string("\t") + extensionName + "\n";
            }
        }

        if (!errlog.empty()) {
            std::string prefix = "Requested device extension not supported:\n";
            throw std::runtime_error(prefix + errlog);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        //deviceFeatures.fillModeNonSolid = VK_TRUE;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
        deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        deviceInfo.ppEnabledExtensionNames = requiredExtensions.data();
        deviceInfo.pEnabledFeatures = &deviceFeatures;

        if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create logical device");
        }

        vkGetDeviceQueue(device, graphicsFamilyIndex, 0, &graphicsQueue);
        vkGetDeviceQueue(device, presentFamilyIndex, 0, &presentQueue);

        VmaVulkanFunctions vulkanFunctions{};
        vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorInfo{};
        allocatorInfo.instance = instance;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.pVulkanFunctions = &vulkanFunctions;
        if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VMA allocator");
        }
    }

    void createSwapchain()
    {
        VkSurfaceFormatKHR surfaceFormat;
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // VSync
        //VkPresentModeKHR presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR; // Unlimited fps
        VkSurfaceCapabilitiesKHR capabilities;

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

        surfaceFormat = formats[0];
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                surfaceFormat = format;
                break;
            }
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D extent = { (uint32_t)width, (uint32_t)height };
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);
        extent.width = glm::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = glm::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }

        uint32_t familyIndices[] = { (uint32_t)graphicsFamilyIndex, (uint32_t)presentFamilyIndex };

        VkSwapchainCreateInfoKHR swapchainInfo{};
        swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainInfo.surface = surface;
        swapchainInfo.minImageCount = imageCount;
        swapchainInfo.imageFormat = surfaceFormat.format;
        swapchainInfo.imageColorSpace = surfaceFormat.colorSpace;
        swapchainInfo.imageExtent = extent;
        swapchainInfo.imageArrayLayers = 1;
        swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = familyIndices;
        swapchainInfo.preTransform = capabilities.currentTransform;
        swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchainInfo.presentMode = presentMode;
        swapchainInfo.clipped = VK_TRUE;
        swapchainInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

        swapchainImageFormat = surfaceFormat.format;
        swapchainExtent = extent;

        swapchainImageViews.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); i++) {
            VkImageViewCreateInfo imageViewInfo{};
            imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            imageViewInfo.image = swapchainImages[i];
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewInfo.format = swapchainImageFormat;
            imageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageViewInfo.subresourceRange.baseMipLevel = 0;
            imageViewInfo.subresourceRange.levelCount = 1;
            imageViewInfo.subresourceRange.baseArrayLayer = 0;
            imageViewInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &imageViewInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views");
            }
        }
    }

    void createDepthImage()
    {
        VkImageCreateInfo depthImageInfo{};
        depthImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
        depthImageInfo.extent.width = swapchainExtent.width;
        depthImageInfo.extent.height = swapchainExtent.height;
        depthImageInfo.extent.depth = 1;
        depthImageInfo.mipLevels = 1;
        depthImageInfo.arrayLayers = 1;
        depthImageInfo.format = VK_FORMAT_D32_SFLOAT;
        depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

        if (vmaCreateImage(allocator, &depthImageInfo, &allocInfo, &depthImage, &depthImageAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create depth image");
        }

        VkImageViewCreateInfo depthImageViewInfo{};
        depthImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthImageViewInfo.image = depthImage;
        depthImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthImageViewInfo.format = VK_FORMAT_D32_SFLOAT;
        depthImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthImageViewInfo.subresourceRange.baseMipLevel = 0;
        depthImageViewInfo.subresourceRange.levelCount = 1;
        depthImageViewInfo.subresourceRange.baseArrayLayer = 0;
        depthImageViewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &depthImageViewInfo, nullptr, &depthImageView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create depth image view");
        }
    }

    void createPipelines()
    {
        VkAttachmentDescription colorAttachmentDesc{};
        colorAttachmentDesc.format = swapchainImageFormat;
        colorAttachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachmentDesc{};
        depthAttachmentDesc.format = VK_FORMAT_D32_SFLOAT;
        depthAttachmentDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachmentDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachmentDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachmentDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachmentDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachmentDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachmentDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        const std::vector<VkAttachmentDescription> attachments = { colorAttachmentDesc, depthAttachmentDesc };

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDesc{};
        subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDesc.colorAttachmentCount = 1;
        subpassDesc.pColorAttachments = &colorAttachmentRef;
        subpassDesc.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = (uint32_t)attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDesc;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass");
        }

        const std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
        };
        VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateInfo.dynamicStateCount = (uint32_t)dynamicStates.size();
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
        inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportStateInfo{};
        viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateInfo.viewportCount = 1;
        viewportStateInfo.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizerInfo{};
        rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizerInfo.depthClampEnable = VK_FALSE;
        rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
        rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
        //rasterizerInfo.polygonMode = VK_POLYGON_MODE_LINE;
        rasterizerInfo.lineWidth = 1.0f;
        rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizerInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // through vulkan recommends clockwise, but counter is more common.
        rasterizerInfo.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisamplingInfo{};
        multisamplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisamplingInfo.sampleShadingEnable = VK_FALSE;
        multisamplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
        depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencilInfo.depthTestEnable = VK_TRUE;
        depthStencilInfo.depthWriteEnable = VK_TRUE;
        depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilInfo.depthBoundsTestEnable = VK_FALSE;
        depthStencilInfo.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachmentInfo{};
        colorBlendAttachmentInfo.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachmentInfo.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlendInfo{};
        colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendInfo.logicOpEnable = VK_FALSE;
        colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendInfo.attachmentCount = 1;
        colorBlendInfo.pAttachments = &colorBlendAttachmentInfo;
        colorBlendInfo.blendConstants[0] = 0.0f;
        colorBlendInfo.blendConstants[1] = 0.0f;
        colorBlendInfo.blendConstants[2] = 0.0f;
        colorBlendInfo.blendConstants[3] = 0.0f;

        std::array<VkDescriptorSetLayoutBinding, 5> layoutBindings{};
        // ubo
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
        // albedo
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        // normal
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        // metallic
        layoutBindings[3].binding = 3;
        layoutBindings[3].descriptorCount = 1;
        layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        // roughness
        layoutBindings[4].binding = 4;
        layoutBindings[4].descriptorCount = 1;
        layoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
        descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutInfo.bindingCount = (uint32_t)layoutBindings.size();
        descriptorSetLayoutInfo.pBindings = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout");
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout");
        }

        // graphics pipeline
        {
            VkShaderModule vertShaderModule;
            VkShaderModule fragShaderModule;
            {
                VkShaderModuleCreateInfo shaderModuleInfo{};
                shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                shaderModuleInfo.codeSize = shader_vert_size;
                shaderModuleInfo.pCode = shader_vert_code;
                if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &vertShaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create vertex shader module");
                }

                shaderModuleInfo.codeSize = shader_frag_size;
                shaderModuleInfo.pCode = shader_frag_code;
                if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &fragShaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create fragment shader module");
                }
            }

            std::array<VkVertexInputBindingDescription, 1> bindingDescription{};
            bindingDescription[0].binding = 0;
            bindingDescription[0].stride = sizeof(Vertex);
            bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
            attributeDescriptions[0].binding = 0;
            attributeDescriptions[0].location = 0;
            attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[0].offset = offsetof(Vertex, position);
            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, normal);
            attributeDescriptions[2].binding = 0;
            attributeDescriptions[2].location = 2;
            attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
            attributeDescriptions[2].offset = offsetof(Vertex, texcoords);

            VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)bindingDescription.size();
            vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
            vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

            VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
            pipelineInfo.pViewportState = &viewportStateInfo;
            pipelineInfo.pRasterizationState = &rasterizerInfo;
            pipelineInfo.pMultisampleState = &multisamplingInfo;
            pipelineInfo.pDepthStencilState = &depthStencilInfo;
            pipelineInfo.pColorBlendState = &colorBlendInfo;
            pipelineInfo.pDynamicState = &dynamicStateInfo;
            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create graphics pipeline");
            }

            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
        }

        // block pipeline
        /*{
            VkShaderModule vertShaderModule;
            VkShaderModule fragShaderModule;
            {
                VkShaderModuleCreateInfo shaderModuleInfo{};
                shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

                shaderModuleInfo.codeSize = block_vert_size;
                shaderModuleInfo.pCode = block_vert_code;
                if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &vertShaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create vertex shader module");
                }

                shaderModuleInfo.codeSize = block_frag_size;
                shaderModuleInfo.pCode = block_frag_code;
                if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &fragShaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("Failed to create fragment shader module");
                }
            }

            VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

            std::array<VkVertexInputBindingDescription, 1> bindingDescription{};
            bindingDescription[0].binding = 0;
            bindingDescription[0].stride = sizeof(glm::ivec4);
            bindingDescription[0].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

            std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
            attributeDescriptions[0].binding = 0;
            attributeDescriptions[0].location = 0;
            attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SINT;
            attributeDescriptions[0].offset = 0;
            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32_SINT;
            attributeDescriptions[1].offset = sizeof(glm::ivec3);

            VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = (uint32_t)bindingDescription.size();
            vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
            vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
            vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
            pipelineInfo.pViewportState = &viewportStateInfo;
            pipelineInfo.pRasterizationState = &rasterizerInfo;
            pipelineInfo.pMultisampleState = &multisamplingInfo;
            pipelineInfo.pDepthStencilState = &depthStencilInfo;
            pipelineInfo.pColorBlendState = &colorBlendInfo;
            pipelineInfo.pDynamicState = &dynamicStateInfo;
            pipelineInfo.layout = pipelineLayout;
            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &blockPipeline) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create block pipeline");
            }

            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
        }*/
    }

    void createFramebuffers()
    {
        swapchainFramebuffers.resize(swapchainImageViews.size());
        for (size_t i = 0; i < swapchainImageViews.size(); i++) {
            const std::vector<VkImageView> attachments = { swapchainImageViews[i], depthImageView };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = (uint32_t)attachments.size();
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapchainExtent.width;
            framebufferInfo.height = swapchainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapchainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create framebuffer");
            }
        }
    }

    void createCommandPool()
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsFamilyIndex;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create command pool");
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate command buffer");
        }

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create synchronization objects for a frame");
        }
    }

    void createBuffers()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "src/models/M1A1_Thompson.obj")) {
            throw std::runtime_error(warn + err);
        }

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        std::unordered_map<glm::ivec3, uint32_t> uniqueVertices{};
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2],
                };
                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2],
                };
                vertex.texcoords = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                };

                glm::ivec3 vtx = {
                    index.vertex_index, 
                    index.normal_index, 
                    index.texcoord_index,
                };
                if (uniqueVertices.count(vtx) == 0) {
                    uniqueVertices[vtx] = (uint32_t)vertices.size();
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vtx]);
            }
        }
        indexCount = (int)indices.size();

        int texChannels;
        int albedoWidth, albedoHeight;
        int normalWidth, normalHeight;
        int metallicWidth, metallicHeight;
        int roughnessWidth, roughnessHeight;
        stbi_uc* albedoPixels = stbi_load("src/models/M1A1_Thompson_Albedo_1K.png", &albedoWidth, &albedoHeight, &texChannels, STBI_rgb_alpha);
        stbi_uc* normalPixels = stbi_load("src/models/M1A1_Thompson_Normal_1K.png", &normalWidth, &normalHeight, &texChannels, STBI_rgb_alpha);
        stbi_uc* metallicPixels = stbi_load("src/models/M1A1_Thompson_Metallic_1K.png", &metallicWidth, &metallicHeight, &texChannels, STBI_rgb_alpha);
        stbi_uc* roughnessPixels = stbi_load("src/models/M1A1_Thompson_Roughness_1K.png", &roughnessWidth, &roughnessHeight, &texChannels, STBI_rgb_alpha);
        if (!albedoPixels || !normalPixels || !metallicPixels || !roughnessPixels) {
            throw std::runtime_error("Failed load textures");
        }

        VkBufferCreateInfo vertexBufferInfo{};
        vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vertexBufferInfo.size = sizeof_container(vertices);
        vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateBuffer(allocator, &vertexBufferInfo, &allocInfo, &vertexBuffer, &vertexAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create vertex buffer");
        }

        VkBufferCreateInfo indexBufferInfo{};
        indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indexBufferInfo.size = sizeof_container(indices);
        indexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateBuffer(allocator, &indexBufferInfo, &allocInfo, &indexBuffer, &indexAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create index buffer");
        }

        auto createTexture = [&](int width, int height, VkImage* image, VmaAllocation* imageAlloc, VkImageView* imageView, VkSampler* sampler) {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = (uint32_t)width;
            imageInfo.extent.height = (uint32_t)height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.flags = 0;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

            if (vmaCreateImage(allocator, &imageInfo, &allocInfo, image, imageAlloc, nullptr) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image");
            }

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = *image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &viewInfo, nullptr, imageView) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image view");
            }

            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physicalDevice, &properties);

            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            //samplerInfo.anisotropyEnable = VK_TRUE;
            samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
            samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.mipLodBias = 0.0f;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 0.0f;

            if (vkCreateSampler(device, &samplerInfo, nullptr, sampler) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create sampler");
            }
        };

        createTexture(albedoWidth, albedoHeight, &albedoMap, &albedoAlloc, &albedoView, &albedoSampler);
        createTexture(normalWidth, normalHeight, &normalMap, &normalAlloc, &normalView, &normalSampler);
        createTexture(metallicWidth, metallicHeight, &metallicMap, &metallicAlloc, &metallicView, &metallicSampler);
        createTexture(roughnessWidth, roughnessHeight, &roughnessMap, &roughnessAlloc, &roughnessView, &roughnessSampler);

        /*std::vector<glm::ivec4> blocks;
        for (int z = 0; z < 128; z++) {
        for (int x = 0; x < 128; x++) {
            float n = fbm(glm::vec2(x,z)/128.0f) * 32.0f;
            float y = glm::round(n);
            float h = glm::round(y/32.0f * 255.0f);
            blocks.push_back(glm::ivec4(x, int(y), z, (int(h)<<8) + 0b111111));
        } }
        blockCount = (int)blocks.size();

        VkBufferCreateInfo blockBufferInfo{};
        blockBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        blockBufferInfo.size = sizeof_container(blocks);
        blockBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        blockBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateBuffer(allocator, &blockBufferInfo, &allocInfo, &blockBuffer, &blockAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create block buffer");
        }*/

        VkBufferCreateInfo uniformBufferInfo{};
        uniformBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        uniformBufferInfo.size = sizeof(UniformBlock);
        uniformBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        uniformBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        if (vmaCreateBuffer(allocator, &uniformBufferInfo, &allocInfo, &uniformBuffer, &uniformAlloc, &uniformInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create uniform buffer");
        }

        VkBuffer stagingBuffer;
        VmaAllocation stagingAlloc;
        uint32_t stagingSize = 32 * 1024 * 1024;

        VkBufferCreateInfo stagingBufferInfo{};
        stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferInfo.size = stagingSize;
        stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        if (vmaCreateBuffer(allocator, &stagingBufferInfo, &allocInfo, &stagingBuffer, &stagingAlloc, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create staging buffer");
        }

        auto beginSingleTimeCommands = [&]() {
            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);

            return commandBuffer;
        };

        auto endSingleTimeCommands = [&](VkCommandBuffer commandBuffer) {
            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);

            vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        };

        auto uploadData = [&](VkBuffer targetBuffer, const void* src_data, uint32_t size) {
            void* data;
            vmaMapMemory(allocator, stagingAlloc, &data);
            memcpy(data, (char*)src_data, size);
            vmaUnmapMemory(allocator, stagingAlloc);

            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkBufferCopy copyRegion{};
            copyRegion.size = size;
            copyRegion.dstOffset = 0;
            vkCmdCopyBuffer(commandBuffer, stagingBuffer, targetBuffer, 1, &copyRegion);

            endSingleTimeCommands(commandBuffer);
        };

        auto transitionImageLayout = [&](VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags sourceStage;
            VkPipelineStageFlags destinationStage;

            if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

                sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            } else {
                throw std::invalid_argument("unsupported layout transition!");
            }

            vkCmdPipelineBarrier(
                commandBuffer,
                sourceStage, destinationStage,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            endSingleTimeCommands(commandBuffer);
        };

        auto uploadImage = [&](VkImage targetImage, const void* src_data, int width, int height, int pixelSize) {
            transitionImageLayout(targetImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            void* data;
            vmaMapMemory(allocator, stagingAlloc, &data);
            memcpy(data, (char*)src_data, width * height * pixelSize);
            vmaUnmapMemory(allocator, stagingAlloc);

            VkCommandBuffer commandBuffer = beginSingleTimeCommands();

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = { (uint32_t)width, (uint32_t)height, 1 };
            vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, targetImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            endSingleTimeCommands(commandBuffer);

            transitionImageLayout(targetImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };

        uploadData(vertexBuffer, vertices.data(), sizeof_container(vertices));
        uploadData(indexBuffer, indices.data(), sizeof_container(indices));
        uploadImage(albedoMap, albedoPixels, albedoWidth, albedoHeight, 4);
        uploadImage(normalMap, normalPixels, normalWidth, normalHeight, 4);
        uploadImage(metallicMap, metallicPixels, metallicWidth, metallicHeight, 4);
        uploadImage(roughnessMap, roughnessPixels, roughnessWidth, roughnessHeight, 4);
        //uploadData(blockBuffer, blocks.data(), sizeof_container(blocks));

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vmaFreeMemory(allocator, stagingAlloc);
        stbi_image_free(albedoPixels);
        stbi_image_free(normalPixels);
        stbi_image_free(metallicPixels);
        stbi_image_free(roughnessPixels);
    }

    void createDescriptors()
    {
        std::array<VkDescriptorPoolSize, 5> poolSizes{};
        // ubo
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 1;
        // albedo
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = 1;
        // normal
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = 1;
        // metallic
        poolSizes[3].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[3].descriptorCount = 1;
        // roughness
        poolSizes[4].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[4].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBlock);

        VkDescriptorImageInfo albedoImageInfo{};
        albedoImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        albedoImageInfo.imageView = albedoView;
        albedoImageInfo.sampler = albedoSampler;

        VkDescriptorImageInfo normalImageInfo{};
        normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        normalImageInfo.imageView = normalView;
        normalImageInfo.sampler = normalSampler;

        VkDescriptorImageInfo metallicImageInfo{};
        metallicImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        metallicImageInfo.imageView = metallicView;
        metallicImageInfo.sampler = metallicSampler;

        VkDescriptorImageInfo roughnessImageInfo{};
        roughnessImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        roughnessImageInfo.imageView = roughnessView;
        roughnessImageInfo.sampler = roughnessSampler;

        std::array<VkWriteDescriptorSet, 5> descriptorWrites{};
        // ubo
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        // albedo
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &albedoImageInfo;
        // normal
        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pImageInfo = &normalImageInfo;
        // metallic
        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = descriptorSet;
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pImageInfo = &metallicImageInfo;
        // roughness
        descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[4].dstSet = descriptorSet;
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pImageInfo = &roughnessImageInfo;

        vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    void recreateSwapchain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        for (auto framebuffer : swapchainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vmaFreeMemory(allocator, depthImageAlloc);
        for (auto imageView : swapchainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);

        createSwapchain();
        createDepthImage();
        createFramebuffers();
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { 0.5f, 0.5f, 1.0f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapchainExtent;
        renderPassInfo.clearValueCount = (uint32_t)clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = (float)swapchainExtent.height;
            viewport.width = (float)swapchainExtent.width;
            viewport.height = -(float)swapchainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapchainExtent;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

            UniformBlock ublock;
            glm::mat4 model = glm::scale(glm::vec3(5.0f));//glm::rotate((float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 projectionView = camera.update(window);
            glm::mat4 MVP = projectionView * model;
            
            ublock.MVP = MVP;
            ublock.uModel = model;
            ublock.viewPos = camera.position;
            memcpy(uniformInfo.pMappedData, &ublock, sizeof(ublock));
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkBuffer vertexBuffers[] = { vertexBuffer };
            VkDeviceSize vertexOffsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);

            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);


            /*ublock.MVP = projectionView;
            ublock.uModel = glm::mat4(1.0f);
            ublock.viewPos = camera.position;
            memcpy(uniformInfo.pMappedData, &ublock, sizeof(ublock));
            vmaFlushAllocation(allocator, uniformAlloc, 0, VK_WHOLE_SIZE);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, blockPipeline);

            VkBuffer blockBuffers[] = { blockBuffer };
            VkDeviceSize blockOffsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, blockBuffers, blockOffsets);

            vkCmdDraw(commandBuffer, 36, blockCount, 0, 0);*/

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }
    }

    void run()
    {
        glfwShowWindow(window);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window)) {
            glfwPollEvents();

            vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

            VkResult result;
            uint32_t imageIndex;
            result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchain();
                continue;
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
                throw std::runtime_error("Failed to acquire swap chain image");
            }

            vkResetFences(device, 1, &inFlightFence);

            vkResetCommandBuffer(commandBuffer, 0);
            recordCommandBuffer(commandBuffer, imageIndex);

            VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
                throw std::runtime_error("Failed to submit draw command buffer");
            }

            VkSwapchainKHR swapchains[] = { swapchain };

            VkPresentInfoKHR presentInfo{};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains;
            presentInfo.pImageIndices = &imageIndex;

            vkQueuePresentKHR(presentQueue, &presentInfo);
        }

        vkDeviceWaitIdle(device);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyBuffer(device, uniformBuffer, nullptr);
        vmaFreeMemory(allocator, uniformAlloc);

        //vkDestroyBuffer(device, blockBuffer, nullptr);
        //vmaFreeMemory(allocator, blockAlloc);

        vkDestroySampler(device, roughnessSampler, nullptr);
        vkDestroyImageView(device, roughnessView, nullptr);
        vkDestroyImage(device, roughnessMap, nullptr);
        vmaFreeMemory(allocator, roughnessAlloc);
        vkDestroySampler(device, metallicSampler, nullptr);
        vkDestroyImageView(device, metallicView, nullptr);
        vkDestroyImage(device, metallicMap, nullptr);
        vmaFreeMemory(allocator, metallicAlloc);
        vkDestroySampler(device, normalSampler, nullptr);
        vkDestroyImageView(device, normalView, nullptr);
        vkDestroyImage(device, normalMap, nullptr);
        vmaFreeMemory(allocator, normalAlloc);
        vkDestroySampler(device, albedoSampler, nullptr);
        vkDestroyImageView(device, albedoView, nullptr);
        vkDestroyImage(device, albedoMap, nullptr);
        vmaFreeMemory(allocator, albedoAlloc);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vmaFreeMemory(allocator, indexAlloc);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vmaFreeMemory(allocator, vertexAlloc);

        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);
        for (auto framebuffer : swapchainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        //vkDestroyPipeline(device, blockPipeline, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vmaFreeMemory(allocator, depthImageAlloc);
        for (auto imageView : swapchainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        vmaDestroyAllocator(allocator);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};

int main()
{
    glfwInit();
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // do not create a default OpenGL context

    constexpr int width = 1600, height = 900;
    GLFWwindow* window = glfwCreateWindow(width, height, "Freecam", nullptr, nullptr);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	glfwSetWindowPos(window, (mode->width - width)/2, (mode->height - height)/2);

    try {
        if (volkInitialize() != VK_SUCCESS) {
            throw std::runtime_error("Failed to find Vulkan on your computer");
        }
        Application application(window);
        application.run();
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
