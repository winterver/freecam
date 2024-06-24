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

const int faces[6][6] = {
    { 5, 1, 3, 5, 3, 7 }, // +x
    { 0, 4, 6, 0, 6, 2 }, // -x
    { 4, 5, 7, 4, 7, 6 }, // +z
    { 1, 0, 2, 1, 2, 3 }, // -z
    { 6, 7, 3, 6, 3, 2 }, // +y
    { 0, 1, 5, 0, 5, 4 }, // -y
};

const vec2 uvs[6] = {
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 1),
    vec2(1, 0),
    vec2(0, 1),
    vec2(0, 0),
};

void main() {
    int face = gl_VertexIndex / 6;
    int index = gl_VertexIndex % 6;
    if (!bool(position.w & (1 << face))) {
        return;
    }
    gl_Position = MVP * vec4(position.xyz+vertices[faces[face][index]], 1.0);
    texCoord = uvs[index];
}
