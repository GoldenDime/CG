#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 counts;
} scene;

// Набор 1 (материал): динамический uniform буфер, сдвиг до свойств конкретной
// модели задаётся при привязке набора дескрипторов
layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 tint;          // rgb: множитель альбедо
	vec4 specular;      // rgb: цвет блика, w: коэффициент блеска
	vec4 uv_transform;  // xy: масштаб, zw: сдвиг текстурных координат
	vec4 params;        // x: яркость emissive
} model;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;

void main() {
	vec4 world_position = model.model * vec4(in_position, 1.0);
	out_world_position = world_position.xyz;
	out_normal = mat3(model.normal_matrix) * in_normal;
	out_uv = in_uv * model.uv_transform.xy + model.uv_transform.zw;
	gl_Position = scene.view_projection * world_position;
}
