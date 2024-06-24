#version 450

layout(location = 0) in ivec4 position;
layout(location = 0) out vec2 texCoord;

layout(push_constant) uniform constants {
    mat4 MVP;
};

const vec3 vertices[8] = {
    vec3(0, 0, 0),
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(1, 1, 0),
    vec3(0, 0, 1),
    vec3(1, 0, 1),
    vec3(0, 1, 1),
    vec3(1, 1, 1),
};

const int faces[6][4] = {
    { 5, 1, 3, 7 }, // +x
    { 0, 4, 6, 2 }, // -x
    { 4, 5, 7, 6 }, // +z
    { 1, 0, 2, 3 }, // -z
    { 6, 7, 3, 2 }, // +y
    { 0, 1, 5, 4 }, // -y
};

const vec2 uvs[4] = {
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 1),
    vec2(0, 0),
};

void main() {
    gl_Position = MVP * vec4(position.xyz+vertices[faces[position.w][gl_VertexIndex]], 1.0);
    texCoord = uvs[gl_VertexIndex];
}