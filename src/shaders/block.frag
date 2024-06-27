#version 450

layout(location = 0) in vec2 texCoord;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec3 vertColor;
layout(location = 3) in vec3 position;

layout(location = 0) out vec4 outColor;

const vec3 lightPos = vec3(64, 50, 64);

layout(push_constant) uniform constants {
    layout(offset = 64) vec3 viewPos;
};

void main() {
    vec3 color = vertColor;
    // ambient
    vec3 ambient = 0.5 * color;
    // diffuse
    vec3 lightDir = normalize(lightPos - position);
    vec3 normal = normalize(Normal);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * color;
    // specular
    vec3 viewDir = normalize(viewPos - position);
    vec3 reflectDir = reflect(-lightDir, normal);
    // blinn
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0);

    vec3 specular = vec3(0.3) * spec; // assuming bright white light color
    outColor = vec4(ambient + diffuse + specular, 1.0);
}
