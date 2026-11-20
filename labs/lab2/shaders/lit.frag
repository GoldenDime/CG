#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;        // rgb: цвет * интенсивность рассеянного света
	vec4 sun_direction;  // xyz: направление НА источник (нормализованное)
	vec4 sun_color;      // rgb: цвет * интенсивность
	uvec4 counts;        // x: число точечных, y: число прожекторных источников
} scene;

struct PointLight {
	vec4 position;  // xyz
	vec4 color;     // rgb: цвет * интенсивность
};

struct SpotLight {
	vec4 position;  // xyz
	vec4 direction; // xyz: направление луча (нормализованное)
	vec4 color;     // rgb: цвет * интенсивность
	vec4 angles;    // x: cos(внутренний угол), y: cos(внешний угол)
};

// Storage буферы: размер массивов не фиксируется в шейдере
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;    // rgb: альбедо
	vec4 specular;  // rgb: цвет блика, w: коэффициент блеска
} model;

// Модель Блинн-Фонга для одного источника.
// L - направление на источник, radiance - пришедший от источника свет.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
	float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	vec3 H = normalize(L + V);
	float specular = pow(max(dot(N, H), 0.0), model.specular.w);

	return radiance * (model.albedo.rgb * n_dot_l + model.specular.rgb * specular);
}

// Затухание по закону обратных квадратов
float attenuation(float distance_squared) {
	return 1.0 / max(distance_squared, 1e-4);
}

void main() {
	vec3 N = normalize(in_normal);
	vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	vec3 color = scene.ambient.rgb * model.albedo.rgb;

	// Направленный источник
	color += blinnPhong(N, V, scene.sun_direction.xyz, scene.sun_color.rgb);

	// Точечные источники
	for (uint i = 0; i < scene.counts.x; ++i) {
		vec3 to_light = point_lights[i].position.xyz - in_world_position;
		float distance_squared = dot(to_light, to_light);
		vec3 L = to_light * inversesqrt(distance_squared);

		color += blinnPhong(N, V, L, point_lights[i].color.rgb * attenuation(distance_squared));
	}

	// Прожекторные источники
	for (uint i = 0; i < scene.counts.y; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance_squared = dot(to_light, to_light);
		vec3 L = to_light * inversesqrt(distance_squared);

		// Мягкий край: плавный переход между внутренним и внешним конусом
		float cos_theta = dot(-L, light.direction.xyz);
		float cos_inner = light.angles.x;
		float cos_outer = light.angles.y;
		float cone = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
		cone = cone * cone * (3.0 - 2.0 * cone);

		if (cone > 0.0) {
			color += blinnPhong(N, V, L, light.color.rgb * attenuation(distance_squared) * cone);
		}
	}

	out_color = vec4(color, 1.0);
}
