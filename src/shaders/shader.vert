#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

layout(location = 0) out vec3 WorldPos;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 TexCoords;

layout(binding = 0, std140) uniform constants {
    mat4 MVP;
    mat4 uModel;
};

void main() {
    gl_Position = MVP * vec4(aPosition, 1.0);
    WorldPos = vec3(uModel * vec4(aPosition, 1));
    Normal = mat3(uModel) * aNormal;                // translation is stored in the 4th row, ignore it, normal doesn't need it.
    TexCoords = aTexCoords;                         // conversion from OpenGL texcoords to Vulkan texcoords is done during model loading.
}
