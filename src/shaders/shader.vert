#version 450

layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec3 iNormal;

layout(location = 0) out vec3 oVertColor;

layout(push_constant) uniform constants {
    mat4 MVP;
};

void main() {
    gl_Position = MVP * vec4(iPosition, 1.0);
    oVertColor = iNormal;
}
