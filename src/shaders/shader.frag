#version 450

layout(location = 0) in vec3 iVertColor;
layout(location = 0) out vec4 FragColor;

void main() {
    FragColor = vec4(iVertColor, 1.0);
}
