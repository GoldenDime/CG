#version 450

// Проход генерации карты теней: только вершинный шейдер, который проецирует
// геометрию в пространство источника света. Фрагментного шейдера нет —
// в карту теней записывается только глубина.

layout(location = 0) in vec3 in_position;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 tint;
	vec4 specular;
	vec4 uv_transform;
	vec4 params;
} model;

layout(push_constant) uniform Light {
	mat4 view_projection; // матрица источника (солнце: ortho * lookAt, прожектор: perspective * lookAt)
} light;

void main() {
	gl_Position = light.view_projection * model.model * vec4(in_position, 1.0);
}
