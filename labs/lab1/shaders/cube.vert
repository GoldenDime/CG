#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Свойства объекта: у каждого объекта свой uniform буфер и свой набор дескрипторов
layout(set = 0, binding = 0) uniform ObjectUniforms {
	mat4 mvp;       // projection * view * model
	vec4 color;     // цвет из интерфейса
	vec4 params;    // x: 1 - умножать на процедурный цвет вершин, 0 - нет
} object;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = object.mvp * vec4(in_position, 1.0);

	vec3 vertex_color = mix(vec3(1.0), in_color, object.params.x);
	out_color = vertex_color * object.color.rgb;
}
