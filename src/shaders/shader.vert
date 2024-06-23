#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform constants {
    mat4 MVP;
};

void main() {
    gl_Position = MVP * vec4(inPosition, 1.0);
    fragColor = inNormal;
}
