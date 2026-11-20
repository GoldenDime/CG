#version 450

// Визуализация источников света: вершины уже в мировых координатах,
// цвет постоянный и не участвует в расчёте освещения

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 counts;
} scene;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = scene.view_projection * vec4(in_position, 1.0);
	out_color = in_color;
}
