#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Набор 0: данные сцены, общие для всех объектов
layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 counts;
} scene;

// Набор 1: данные конкретной модели
layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} model;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_normal;

void main() {
	vec4 world_position = model.model * vec4(in_position, 1.0);
	out_world_position = world_position.xyz;
	out_normal = mat3(model.normal_matrix) * in_normal;
	gl_Position = scene.view_projection * world_position;
}
