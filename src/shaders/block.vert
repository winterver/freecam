#version 450

layout(location = 0) in ivec3 iPosition;
layout(location = 1) in int iFace;

layout(location = 0) out vec2 oTexCoord;
layout(location = 1) out vec3 oNormal;
layout(location = 2) out vec3 oVertColor;
layout(location = 3) out vec3 oPosition;

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

const int faces[36] = {
    5, 1, 3, 5, 3, 7, // +x
    0, 4, 6, 0, 6, 2, // -x
    4, 5, 7, 4, 7, 6, // +z
    1, 0, 2, 1, 2, 3, // -z
    6, 7, 3, 6, 3, 2, // +y
    0, 1, 5, 0, 5, 4, // -y
};

const vec2 uvs[6] = {
    vec2(1, 0),
    vec2(1, 1),
    vec2(0, 1),
    vec2(1, 0),
    vec2(0, 1),
    vec2(0, 0),
};

const vec3 normals[6] = {
    vec3(1, 0, 0),
    vec3(-1, 0, 0),
    vec3(0, 0, 1),
    vec3(0, 0, -1),
    vec3(0, 1, 0),
    vec3(0, -1, 0),
};

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    if (!bool(iFace & (1 << (gl_VertexIndex/6)))) {
        return;
    }

    vec3 position = iPosition + vertices[faces[gl_VertexIndex]];
    gl_Position = MVP * vec4(position, 1.0);

    oTexCoord = uvs[gl_VertexIndex % 6];
    oNormal = normals[gl_VertexIndex / 6];
    float H = float(iFace>>8)/255.0f;
    oVertColor = hsv2rgb(vec3(H, 1.0, 0.3));
    oPosition = position;
}
