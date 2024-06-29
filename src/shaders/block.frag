#version 450

layout(location = 0) in vec2 iTexCoord;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec3 iVertColor;
layout(location = 3) in vec3 iPosition;

layout(location = 0) out vec4 FragColor;

layout(push_constant) uniform constants {
    layout(offset = 64) vec3 viewPos;
};

const vec3 light = vec3(1.0, 1.0, 1.0);
const vec3 lightPos = vec3(64, 50, 64);

void main() {
    vec3 color = iVertColor;
    vec3 normal = normalize(iNormal);

    // ambient
    vec3 ambient = 0.5 * light * color;

    // diffuse
    vec3 lightDir = normalize(lightPos - iPosition);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * light * color;

    // blinn specular
    vec3 viewDir = normalize(viewPos - iPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    vec3 specular = spec * light;
                                      
    FragColor = vec4(ambient + diffuse /*+ specular*/, 1.0);
}
