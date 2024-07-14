#version 450

layout(location = 0) in vec3 WorldPos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoords;

layout(location = 0) out vec4 FragColor;

layout(binding = 0, std140) uniform constants {
    layout(offset = 128) vec3 viewPos;
};

layout(binding = 1) uniform sampler2D albedoMap;
layout(binding = 2) uniform sampler2D normalMap;
layout(binding = 3) uniform sampler2D metallicMap;
layout(binding = 4) uniform sampler2D roughnessMap;

vec3 materialcolor()
{
    return texture(albedoMap, TexCoords).rgb;
}

vec3 computeTBN()
{
    vec3 tangentNormal = texture(normalMap, TexCoords).xyz * 2.0 - 1.0;

    vec3 Q1  = dFdx(WorldPos);
    vec3 Q2  = dFdy(WorldPos);
    vec2 st1 = dFdx(TexCoords);
    vec2 st2 = dFdy(TexCoords);

    vec3 N   = normalize(Normal);
    vec3 T  = normalize(Q1*st2.t - Q2*st1.t);
    vec3 B  = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

const float PI = 3.14159265359;

float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom); 
}

float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

vec3 F_Schlick(float cosTheta, float metallic)
{
	vec3 F0 = mix(vec3(0.04), materialcolor(), metallic); // * material.specular
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); 
	return F;    
}

vec3 BRDF(vec3 L, vec3 V, vec3 N, float metallic, float roughness)
{
	// Precalculate vectors and dot products	
	vec3 H = normalize (V + L);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float dotLH = clamp(dot(L, H), 0.0, 1.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);

	// Light color fixed
	vec3 lightColor = vec3(1.0);

	vec3 color = vec3(0);

	if (dotNL > 0.0)
	{
		float R = max(0.05, roughness);
		float D = D_GGX(dotNH, R); 
		float G = G_SchlicksmithGGX(dotNL, dotNV, R);
		vec3 F = F_Schlick(dotNV, metallic);

		vec3 spec = D * F * G / (4.0 * dotNL * dotNV);

		color += spec * dotNL * lightColor;
	}

	return color;
}

void main()
{
    vec3 N = computeTBN();
	vec3 V = normalize(viewPos - WorldPos);
	float M = texture(metallicMap, TexCoords).r;
	float R = texture(roughnessMap, TexCoords).r;

    #define NUM_LIGHTS 3
    vec3 lightPos[] = {
        viewPos,
        vec3(0.0f, 0.0f, 5.0f),
        vec3(10.0f, 0.0f, 0.0f),
    };

	vec3 Lo = vec3(0.0);
	for (int i = 0; i < NUM_LIGHTS; i++) {
	  vec3 L = normalize(lightPos[i] - WorldPos);
	  //vec3 L = normalize(vec3(10, 10, 0));
	  Lo += BRDF(L, V, N, M, R);
	}

	Lo += materialcolor() * 0.03;

	FragColor = vec4(pow(Lo, vec3(1.0/2.2)), 1);
}
