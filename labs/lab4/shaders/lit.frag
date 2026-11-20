#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 counts;
	mat4 sun_view_projection;  // ortho * lookAt солнца
	mat4 spot_view_projection; // perspective * lookAt прожектора, отбрасывающего тень
	vec4 shadow_params;        // x: радиус PCF, y: тени от солнца вкл., z: индекс прожектора с тенью (-1 — нет)
	vec4 shadow_texel;         // xy: размер texel карты солнца, zw: карты прожектора
} scene;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 angles;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

// Карты теней сэмплируются через сэмплер со сравнением глубины (оператор «меньше»):
// texture() возвращает долю texel'ей, для которых глубина фрагмента < глубины в карте
layout(set = 0, binding = 3) uniform sampler2DShadow sun_shadow_map;
layout(set = 0, binding = 4) uniform sampler2DShadow spot_shadow_map;

layout(set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec4 tint;
	vec4 specular;
	vec4 uv_transform;
	vec4 params;
} model;

// Текстуры материала. Если у материала нет specular/emissive текстуры,
// вместо неё привязана служебная текстура из одного чёрного пикселя.
layout(set = 1, binding = 1) uniform sampler2D albedo_map;
layout(set = 1, binding = 2) uniform sampler2D specular_map;
layout(set = 1, binding = 3) uniform sampler2D emissive_map;

// Нетривиальное сэмплирование включается push константами,
// а не полями в uniform буфере модели
layout(push_constant) uniform Effect {
	uint flags;      // бит 0: включить эффект
	float time;
	float speed;
	float strength;  // сила модуляции цвета
} effect;

const uint EFFECT_ENABLED = 1u;

vec3 albedo;
vec3 specular_color;

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
	float n_dot_l = max(dot(N, L), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	vec3 H = normalize(L + V);
	float specular = pow(max(dot(N, H), 0.0), model.specular.w);

	return radiance * (albedo * n_dot_l + specular_color * specular);
}

// Возвращает 1, если точка освещена, 0 — если в тени (с PCF — промежуточные значения)
float shadowFactor(sampler2DShadow shadow_map, mat4 light_view_projection, vec2 texel) {
	vec4 clip = light_view_projection * vec4(in_world_position, 1.0);
	if (clip.w <= 0.0) {
		return 1.0;
	}

	vec3 ndc = clip.xyz / clip.w;
	vec2 uv = ndc.xy * 0.5 + 0.5;
	if (ndc.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
		return 1.0; // за пределами области карты теней
	}

	int radius = int(scene.shadow_params.x);
	float sum = 0.0;
	for (int y = -radius; y <= radius; ++y) {
		for (int x = -radius; x <= radius; ++x) {
			sum += texture(shadow_map, vec3(uv + vec2(x, y) * texel, ndc.z));
		}
	}
	float taps = float((2 * radius + 1) * (2 * radius + 1));
	return sum / taps;
}

float attenuation(float distance_squared) {
	return 1.0 / max(distance_squared, 1e-4);
}

vec2 rotate(vec2 v, float angle) {
	float c = cos(angle);
	float s = sin(angle);
	return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Одна и та же текстура сэмплируется трижды разными текстурными координатами,
// зависящими от времени, а результат модулируется переливающимся цветом.
vec3 sampleAlbedoAnimated(vec2 uv) {
	float t = effect.time * effect.speed;

	vec2 uv_rotating = rotate(uv - 0.5, t * 0.5) + 0.5;
	vec2 uv_scrolling = uv * 2.0 + vec2(t * 0.15, -t * 0.1);
	vec2 uv_wavy = uv + 0.04 * vec2(sin(uv.y * 20.0 + t * 3.0), cos(uv.x * 20.0 + t * 3.0));

	vec3 a = texture(albedo_map, uv_rotating).rgb;
	vec3 b = texture(albedo_map, uv_scrolling).rgb;
	vec3 c = texture(albedo_map, uv_wavy).rgb;

	float blend = 0.5 + 0.5 * sin(t);
	vec3 mixed = mix(a, b, blend) * 0.7 + c * 0.3;

	vec3 modulation = 0.5 + 0.5 * cos(t + uv.xyx * 3.0 + vec3(0.0, 2.094, 4.188));
	return mix(mixed, mixed * modulation * 1.6, effect.strength);
}

void main() {
	if ((effect.flags & EFFECT_ENABLED) != 0u) {
		albedo = sampleAlbedoAnimated(in_uv);
	} else {
		albedo = texture(albedo_map, in_uv).rgb;
	}
	albedo *= model.tint.rgb;

	specular_color = model.specular.rgb * texture(specular_map, in_uv).r;

	vec3 N = normalize(in_normal);
	vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	vec3 color = scene.ambient.rgb * albedo;

	float sun_shadow = scene.shadow_params.y > 0.5
		? shadowFactor(sun_shadow_map, scene.sun_view_projection, scene.shadow_texel.xy)
		: 1.0;
	color += blinnPhong(N, V, scene.sun_direction.xyz, scene.sun_color.rgb) * sun_shadow;

	for (uint i = 0; i < scene.counts.x; ++i) {
		vec3 to_light = point_lights[i].position.xyz - in_world_position;
		float distance_squared = dot(to_light, to_light);
		vec3 L = to_light * inversesqrt(distance_squared);

		color += blinnPhong(N, V, L, point_lights[i].color.rgb * attenuation(distance_squared));
	}

	for (uint i = 0; i < scene.counts.y; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position.xyz - in_world_position;
		float distance_squared = dot(to_light, to_light);
		vec3 L = to_light * inversesqrt(distance_squared);

		float cos_theta = dot(-L, light.direction.xyz);
		float cone = clamp((cos_theta - light.angles.y) / max(light.angles.x - light.angles.y, 1e-4), 0.0, 1.0);
		cone = cone * cone * (3.0 - 2.0 * cone);

		if (cone > 0.0) {
			float shadow = int(scene.shadow_params.z) == int(i)
				? shadowFactor(spot_shadow_map, scene.spot_view_projection, scene.shadow_texel.zw)
				: 1.0;
			color += blinnPhong(N, V, L, light.color.rgb * attenuation(distance_squared) * cone) * shadow;
		}
	}

	// Свечение добавляется после расчёта освещения и не зависит от источников
	color += texture(emissive_map, in_uv).rgb * model.params.x;

	out_color = vec4(color, 1.0);
}
