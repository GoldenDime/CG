// =============================================================================
//  Лабораторная работа №4: Наложение теней
//  (продолжение ЛР №3: текстуры, Блинн-Фонг, точечные и прожекторные источники)
//
//  Базовое задание:
//   - карта теней (глубина с точки зрения солнца) генерируется каждый кадр
//     при помощи динамического рендеринга (vkCmdBeginRendering, Vulkan 1.3),
//     без VkRenderPass и VkFramebuffer
//   - отдельный графический конвейер с единственным вершинным шейдером
//     (shadow.vert), который просто проецирует геометрию
//   - для солнца используется ортографическая проекция и матрица LookAt
//   - новый VkSampler со сравнением глубины (оператор «меньше»), карта теней
//     читается в основном фрагментном шейдере как sampler2DShadow
//   - в интерфейсе: площадь, покрываемая ортографической проекцией, и сдвиг
//     глубины (depth bias)
//  Дополнительное задание:
//   - один из прожекторных источников (выбирается в интерфейсе) тоже
//     отбрасывает тень: вторая карта теней с перспективной проекцией
//
//  Управление камерой: зажать ПКМ и двигать мышь — поворот,
//  WASD — перемещение, Q/E — вниз/вверх, Shift — быстрее.
// =============================================================================

#include "application.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vk_utils.hpp"

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct DebugVertex {
	glm::vec3 position;
	glm::vec3 color;
};

// ---- Структуры, разделяемые с шейдерами -------------------------------------

struct SceneUniforms {
	glm::mat4 view_projection;
	glm::vec4 camera_position;
	glm::vec4 ambient;
	glm::vec4 sun_direction;
	glm::vec4 sun_color;
	glm::uvec4 counts;
	glm::mat4 sun_view_projection;
	glm::mat4 spot_view_projection;
	glm::vec4 shadow_params;
	glm::vec4 shadow_texel;
};

struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 color;
};

struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 color;
	glm::vec4 angles;
};

// Элемент массива в динамическом uniform буфере
struct ModelUniforms {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 tint;
	glm::vec4 specular;
	glm::vec4 uv_transform;
	glm::vec4 params;
};

// Push константы нетривиального сэмплирования
struct EffectPushConstants {
	uint32_t flags;
	float time;
	float speed;
	float strength;
};

constexpr uint32_t effect_enabled = 1u;

// Push константы прохода теней
struct ShadowPushConstants {
	glm::mat4 light_view_projection;
};

// ---- Сцена ------------------------------------------------------------------

enum class Trajectory : int {
	Circle,
	Lissajous,
	Rose,
};

// Текстурный материал: набор текстур и свой набор дескрипторов
struct Material {
	std::string name;
	vkutil::Image albedo;
	vkutil::Image specular;  // может отсутствовать -> чёрная текстура
	vkutil::Image emissive;  // может отсутствовать -> чёрная текстура
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct Object {
	std::string name;

	glm::vec3 position{0.0f};
	glm::vec3 rotation{0.0f};
	glm::vec3 scale{1.0f};

	int material = 0;
	glm::vec3 tint{1.0f};
	glm::vec3 specular{1.0f};
	float shininess = 32.0f;
	glm::vec2 uv_scale{1.0f};
	float emissive_strength = 1.0f;

	bool nontrivial_sampling = false;

	bool animate = false;
	Trajectory trajectory = Trajectory::Lissajous;
	float speed = 1.0f;
	float radius = 2.0f;
	float height = 0.5f;
	float frequency_a = 3.0f;
	float frequency_b = 2.0f;
	glm::vec3 spin{30.0f, 90.0f, 0.0f};
	float time = 0.0f;
};

struct Camera {
	glm::vec3 position{0.0f, 3.0f, 9.0f};
	float yaw = 0.0f;
	float pitch = -15.0f;
	float fov = 60.0f;
	float near_plane = 0.1f;
	float far_plane = 200.0f;
	float move_speed = 5.0f;
	float mouse_sensitivity = 0.15f;

	glm::vec3 forward() const {
		const float y = glm::radians(yaw);
		const float p = glm::radians(pitch);
		return { std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p) };
	}
};

struct DirectionalLight {
	float azimuth = 35.0f;
	float elevation = 50.0f;
	glm::vec3 color{1.0f, 0.95f, 0.85f};
	float intensity = 1.1f;

	glm::vec3 direction() const {
		const float a = glm::radians(azimuth);
		const float e = glm::radians(elevation);
		return { std::cos(e) * std::sin(a), std::sin(e), std::cos(e) * std::cos(a) };
	}
};

struct PointLight {
	glm::vec3 position{0.0f, 2.0f, 0.0f};
	glm::vec3 color{1.0f};
	float intensity = 8.0f;

	bool orbit = false;
	float orbit_radius = 3.0f;
	float orbit_speed = 1.0f;
	float orbit_phase = 0.0f;

	glm::vec3 currentPosition() const {
		if (!orbit) {
			return position;
		}
		return position + glm::vec3(std::cos(orbit_phase), 0.0f, std::sin(orbit_phase)) * orbit_radius;
	}
};

struct SpotLight {
	glm::vec3 position{0.0f, 5.0f, 0.0f};
	float yaw = 0.0f;
	float pitch = -90.0f;
	glm::vec3 color{1.0f};
	float intensity = 30.0f;
	float inner_angle = 15.0f;
	float outer_angle = 25.0f;

	glm::vec3 direction() const {
		const float y = glm::radians(yaw);
		const float p = glm::radians(pitch);
		return { std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p) };
	}
};

// ---- Ресурсы Vulkan ---------------------------------------------------------

constexpr uint32_t max_objects = 256;
constexpr uint32_t max_materials = 32;

VkDescriptorSetLayout scene_set_layout;
VkDescriptorSetLayout material_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_descriptor_set;

VkPipelineLayout lit_pipeline_layout;
VkPipelineLayout unlit_pipeline_layout;
VkPipeline lit_pipeline;
VkPipeline unlit_triangles_pipeline;
VkPipeline unlit_lines_pipeline;

VkSampler texture_sampler;
vkutil::Image black_texture; // служебная текстура 1x1

vkutil::Buffer vertex_buffer; // VRAM
vkutil::Buffer index_buffer;  // VRAM
uint32_t index_count;

vkutil::Buffer scene_uniform_buffer;
vkutil::Buffer model_uniform_buffer; // динамический uniform буфер
VkDeviceSize model_uniform_stride;   // размер элемента с учётом выравнивания
vkutil::Buffer point_lights_buffer;
vkutil::Buffer spot_lights_buffer;
vkutil::Buffer debug_vertex_buffer;

// Карты теней (текстуры глубины), в которые рисуется сцена с точки зрения источников
constexpr VkFormat shadow_map_format = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t sun_shadow_map_size = 2048;
constexpr uint32_t spot_shadow_map_size = 1024;

vkutil::Image sun_shadow_map;
vkutil::Image spot_shadow_map;
VkSampler shadow_sampler;         // сэмплер со сравнением глубины
VkPipelineLayout shadow_pipeline_layout;
VkPipeline shadow_pipeline;       // конвейер для динамического рендеринга, только вершинный шейдер
VkDescriptorSet sun_shadow_preview = VK_NULL_HANDLE;  // для отображения в ImGui
VkDescriptorSet spot_shadow_preview = VK_NULL_HANDLE;

// ---- Состояние --------------------------------------------------------------

std::vector<Material> materials;
std::vector<Object> objects;
int selected_object = 0;
int created_objects = 0;

Camera camera;

glm::vec3 ambient_color{0.6f, 0.7f, 1.0f};
float ambient_intensity = 0.06f;
DirectionalLight sun;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;
int selected_point_light = 0;
int selected_spot_light = 0;

bool show_light_markers = true;
bool animation_playing = true;
double previous_time = 0.0;
float effect_time = 0.0f;
float effect_speed = 1.0f;
float effect_strength = 0.6f;

glm::vec3 clear_color{0.02f, 0.02f, 0.03f};

struct ShadowSettings {
	bool sun_enabled = true;
	glm::vec3 center{0.0f, 0.0f, 0.0f}; // точка, на которую смотрит «камера» солнца
	float area = 13.0f;                 // половина стороны квадрата ортографической проекции
	float distance = 25.0f;             // расстояние от центра до солнца
	float near_plane = 0.1f;
	float far_plane = 60.0f;

	// Сдвиг глубины: константный и зависящий от наклона полигона
	float bias_constant = 1.25f;
	float bias_slope = 1.75f;

	int pcf_radius = 1;                 // 0 — без сглаживания, 1 — 3x3, 2 — 5x5

	int spot_index = 0;                 // прожектор, отбрасывающий тень (-1 — ни один)
	float spot_near = 0.5f;
	float spot_far = 40.0f;
	float spot_bias_constant = 1.25f;
	float spot_bias_slope = 1.75f;

	bool show_preview = true;
} shadows;

// ---- Геометрия --------------------------------------------------------------

// Куб с ребром 1, 24 вершины. Текстурные координаты каждой грани покрывают
// всю текстуру [0; 1] x [0; 1] и генерируются вместе с позицией и нормалью.
void createCubeGeometry() {
	struct Face {
		glm::vec3 n, u, v;
	};

	const Face faces[] = {
		{ { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },
		{ { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
		{ { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, -1 } },
		{ { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
		{ { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },
		{ { 0, 0, -1 }, { -1, 0, 0 }, { 0, 1, 0 } },
	};

	const glm::vec2 corners[] = { { -1, -1 }, { 1, -1 }, { 1, 1 }, { -1, 1 } };

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for (const Face& face : faces) {
		const uint32_t base = uint32_t(vertices.size());
		for (const glm::vec2& c : corners) {
			const glm::vec3 position = 0.5f * (face.n + c.x * face.u + c.y * face.v);
			const glm::vec2 uv = { 0.5f + 0.5f * c.x, 0.5f - 0.5f * c.y };
			vertices.push_back({ position, face.n, uv });
		}
		indices.insert(indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
	}

	index_count = uint32_t(indices.size());

	// Вершины и индексы копируются в VRAM через staging буфер
	vertex_buffer = vkutil::createDeviceBuffer(vertices.data(), sizeof(Vertex) * vertices.size(),
	                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	index_buffer = vkutil::createDeviceBuffer(indices.data(), sizeof(uint32_t) * indices.size(),
	                                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

// ---- Дескрипторы ------------------------------------------------------------

void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           VkBuffer buffer, VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = range,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void writeImageDescriptor(VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler) {
	const VkDescriptorImageInfo image_info = {
		.sampler = sampler,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &image_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

void ensureStorageCapacity(vkutil::Buffer& buffer, uint32_t binding, size_t element_size, size_t count) {
	const VkDeviceSize required = VkDeviceSize(element_size * std::max<size_t>(count, 1));
	if (buffer.buffer != VK_NULL_HANDLE && buffer.size >= required) {
		return;
	}

	VkDeviceSize capacity = buffer.size > 0 ? buffer.size : VkDeviceSize(element_size * 4);
	while (capacity < required) {
		capacity *= 2;
	}

	vkutil::destroyBuffer(buffer);
	buffer = vkutil::createHostBuffer(capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	writeBufferDescriptor(scene_descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      buffer.buffer, VK_WHOLE_SIZE);
}

// Загружает текстуры материала и создаёт для него набор дескрипторов:
// binding 0 — общий динамический uniform буфер моделей, 1-3 — текстуры.
bool addMaterial(const std::string& name, const char* albedo_file, const char* specular_file,
                 const char* emissive_file) {
	auto& context = graphics::internal::context;

	Material material;
	material.name = name;

	if (!vkutil::loadTexture(std::string(ASSETS_DIR "textures/") + albedo_file, true, material.albedo)) {
		return false;
	}
	if (specular_file != nullptr &&
	    !vkutil::loadTexture(std::string(ASSETS_DIR "textures/") + specular_file, false, material.specular)) {
		return false;
	}
	if (emissive_file != nullptr &&
	    !vkutil::loadTexture(std::string(ASSETS_DIR "textures/") + emissive_file, true, material.emissive)) {
		return false;
	}

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &material_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &alloc_info, &material.descriptor_set) != VK_SUCCESS) {
		return false;
	}

	writeBufferDescriptor(material.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
	                      model_uniform_buffer.buffer, sizeof(ModelUniforms));
	writeImageDescriptor(material.descriptor_set, 1, material.albedo.view, texture_sampler);
	writeImageDescriptor(material.descriptor_set, 2,
	                     material.specular.view ? material.specular.view : black_texture.view, texture_sampler);
	writeImageDescriptor(material.descriptor_set, 3,
	                     material.emissive.view ? material.emissive.view : black_texture.view, texture_sampler);

	materials.push_back(std::move(material));
	return true;
}

int findMaterial(const char* name) {
	for (size_t i = 0; i < materials.size(); ++i) {
		if (materials[i].name == name) {
			return int(i);
		}
	}
	return 0;
}

Object& addObject(const std::string& name, const glm::vec3& position, const glm::vec3& scale,
                  const char* material) {
	Object object;
	object.name = name.empty() ? "Cube #" + std::to_string(++created_objects) : name;
	object.position = position;
	object.scale = scale;
	object.material = findMaterial(material);
	objects.push_back(std::move(object));
	return objects.back();
}

// ---- Трансформации ----------------------------------------------------------

glm::vec3 trajectoryOffset(const Object& object) {
	const float t = object.time;
	switch (object.trajectory) {
	case Trajectory::Circle:
		return { object.radius * std::cos(t), 0.0f, object.radius * std::sin(t) };

	case Trajectory::Lissajous:
		return {
			object.radius * std::sin(object.frequency_a * t + float(M_PI) * 0.5f),
			object.height * std::sin((object.frequency_a + object.frequency_b) * t),
			object.radius * std::sin(object.frequency_b * t),
		};

	case Trajectory::Rose: {
		const float r = object.radius * std::cos(object.frequency_a * t);
		return { r * std::cos(t), object.height * std::sin(2.0f * t), r * std::sin(t) };
	}
	}
	return glm::vec3(0.0f);
}

glm::mat4 modelMatrix(const Object& object) {
	glm::vec3 position = object.position;
	glm::vec3 rotation = object.rotation;

	if (object.animate) {
		position += trajectoryOffset(object);
		rotation += object.spin * object.time;
	}

	glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
	model = glm::rotate(model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, object.scale);
	return model;
}

glm::mat4 viewProjectionMatrix() {
	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	glm::mat4 projection = glm::perspective(glm::radians(camera.fov), aspect,
	                                        camera.near_plane, camera.far_plane);
	projection[1][1] *= -1.0f;

	const glm::mat4 view = glm::lookAt(camera.position, camera.position + camera.forward(),
	                                   glm::vec3(0.0f, 1.0f, 0.0f));
	return projection * view;
}

// Солнце: ортографическая проекция квадрата 2*area x 2*area, ориентация — LookAt
glm::mat4 sunViewProjection() {
	const glm::vec3 direction = sun.direction();
	const glm::vec3 up = std::abs(direction.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
	const glm::vec3 eye = shadows.center + direction * shadows.distance;

	const glm::mat4 view = glm::lookAt(eye, shadows.center, up);
	const glm::mat4 projection = glm::ortho(-shadows.area, shadows.area, -shadows.area, shadows.area,
	                                        shadows.near_plane, shadows.far_plane);
	return projection * view;
}

// Прожектор: перспективная проекция с углом обзора, охватывающим внешний конус
glm::mat4 spotViewProjection(const SpotLight& light) {
	const glm::vec3 direction = light.direction();
	const glm::vec3 up = std::abs(direction.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

	const glm::mat4 view = glm::lookAt(light.position, light.position + direction, up);
	const float fov = std::min(2.0f * light.outer_angle + 2.0f, 170.0f);
	const glm::mat4 projection = glm::perspective(glm::radians(fov), 1.0f, shadows.spot_near, shadows.spot_far);
	return projection * view;
}

// ---- Визуализация источников ------------------------------------------------

void appendMarkerCube(std::vector<DebugVertex>& out, const glm::vec3& center, float size,
                      const glm::vec3& color) {
	const glm::vec3 c[8] = {
		center + size * glm::vec3(-1, -1, -1), center + size * glm::vec3(1, -1, -1),
		center + size * glm::vec3(-1, 1, -1), center + size * glm::vec3(1, 1, -1),
		center + size * glm::vec3(-1, -1, 1), center + size * glm::vec3(1, -1, 1),
		center + size * glm::vec3(-1, 1, 1), center + size * glm::vec3(1, 1, 1),
	};
	const int indices[] = {
		0, 4, 6, 0, 6, 2,  1, 3, 7, 1, 7, 5,  0, 1, 5, 0, 5, 4,
		2, 6, 7, 2, 7, 3,  0, 2, 3, 0, 3, 1,  4, 5, 7, 4, 7, 6,
	};
	for (int i : indices) {
		out.push_back({ c[i], color });
	}
}

void appendSpotCone(std::vector<DebugVertex>& out, const SpotLight& light) {
	const glm::vec3 white(1.0f);
	const glm::vec3 grey(0.45f);
	const glm::vec3 dir = light.direction();
	const glm::vec3 helper = std::abs(dir.y) > 0.99f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
	const glm::vec3 right = glm::normalize(glm::cross(dir, helper));
	const glm::vec3 up = glm::cross(right, dir);

	const float length = 2.5f;
	constexpr int segments = 24;

	auto circle = [&](float angle_degrees, const glm::vec3& color, bool rays) {
		const float radius = length * std::tan(glm::radians(angle_degrees));
		const glm::vec3 center = light.position + dir * length;
		for (int i = 0; i < segments; ++i) {
			const float a0 = 2.0f * float(M_PI) * float(i) / segments;
			const float a1 = 2.0f * float(M_PI) * float(i + 1) / segments;
			const glm::vec3 p0 = center + radius * (std::cos(a0) * right + std::sin(a0) * up);
			const glm::vec3 p1 = center + radius * (std::cos(a1) * right + std::sin(a1) * up);
			out.push_back({ p0, color });
			out.push_back({ p1, color });
			if (rays && i % 6 == 0) {
				out.push_back({ light.position, color });
				out.push_back({ p0, color });
			}
		}
	};

	circle(light.outer_angle, white, true);
	circle(light.inner_angle, grey, false);

	out.push_back({ light.position, white });
	out.push_back({ light.position + dir * length, white });
}

// ---- Интерфейс --------------------------------------------------------------

void drawCameraTab() {
	ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
	ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.1f deg");
	ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.1f deg");
	ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f, "%.0f deg");
	ImGui::DragFloatRange2("Near / far", &camera.near_plane, &camera.far_plane, 0.05f, 0.01f, 1000.0f);
	ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 30.0f);
	ImGui::SliderFloat("Mouse sensitivity", &camera.mouse_sensitivity, 0.02f, 0.5f);
	ImGui::Separator();
	ImGui::TextWrapped("Hold RMB and move the mouse to look around. WASD - move, Q/E - down/up, Shift - faster.");
}

void drawPointLightEditor(PointLight& light) {
	ImGui::DragFloat3("Position", &light.position.x, 0.05f);
	ImGui::ColorEdit3("Color", &light.color.x);
	ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
	ImGui::Checkbox("Orbit around position", &light.orbit);
	if (light.orbit) {
		ImGui::SliderFloat("Orbit radius", &light.orbit_radius, 0.0f, 10.0f);
		ImGui::SliderFloat("Orbit speed", &light.orbit_speed, -5.0f, 5.0f);
	}
}

void drawSpotLightEditor(SpotLight& light) {
	ImGui::DragFloat3("Position", &light.position.x, 0.05f);
	ImGui::SliderFloat("Yaw", &light.yaw, -180.0f, 180.0f, "%.1f deg");
	ImGui::SliderFloat("Pitch", &light.pitch, -90.0f, 90.0f, "%.1f deg");
	ImGui::ColorEdit3("Color", &light.color.x);
	ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 300.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, 89.0f, "%.1f deg");
	ImGui::SliderFloat("Outer angle", &light.outer_angle, 0.0f, 89.0f, "%.1f deg");
	light.outer_angle = std::max(light.outer_angle, light.inner_angle);
}

template <typename T, typename Editor>
void drawLightList(const char* label, std::vector<T>& lights, int& selected, const T& prototype,
                   Editor editor) {
	ImGui::PushID(label);

	if (ImGui::Button("Add")) {
		lights.push_back(prototype);
		selected = int(lights.size()) - 1;
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(lights.empty());
	if (ImGui::Button("Remove selected")) {
		lights.erase(lights.begin() + selected);
		selected = std::max(0, std::min(selected, int(lights.size()) - 1));
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::Text("Count: %d", int(lights.size()));

	if (ImGui::BeginListBox("##list", ImVec2(-FLT_MIN, 4 * ImGui::GetTextLineHeightWithSpacing()))) {
		for (int i = 0; i < int(lights.size()); ++i) {
			const std::string name = std::string(label) + " #" + std::to_string(i);
			if (ImGui::Selectable(name.c_str(), selected == i)) {
				selected = i;
			}
		}
		ImGui::EndListBox();
	}

	if (!lights.empty()) {
		editor(lights[size_t(selected)]);
	}

	ImGui::PopID();
}

void drawLightsTab() {
	ImGui::Checkbox("Show light sources", &show_light_markers);

	if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("ambient");
		ImGui::ColorEdit3("Color", &ambient_color.x);
		ImGui::SliderFloat("Intensity", &ambient_intensity, 0.0f, 1.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("sun");
		ImGui::SliderFloat("Azimuth", &sun.azimuth, -180.0f, 180.0f, "%.1f deg");
		ImGui::SliderFloat("Elevation", &sun.elevation, -90.0f, 90.0f, "%.1f deg");
		ImGui::ColorEdit3("Color", &sun.color.x);
		ImGui::SliderFloat("Intensity", &sun.intensity, 0.0f, 5.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Point lights")) {
		PointLight prototype;
		prototype.position = camera.position + camera.forward() * 3.0f;
		drawLightList("Point", point_lights, selected_point_light, prototype, drawPointLightEditor);
	}

	if (ImGui::CollapsingHeader("Spot lights")) {
		SpotLight prototype;
		prototype.position = camera.position;
		prototype.yaw = camera.yaw;
		prototype.pitch = camera.pitch;
		drawLightList("Spot", spot_lights, selected_spot_light, prototype, drawSpotLightEditor);
	}
}

void drawObjectEditor(Object& object) {
	ImGui::PushID(&object);

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", &object.position.x, 0.05f);
	ImGui::DragFloat3("Rotation", &object.rotation.x, 1.0f, -360.0f, 360.0f, "%.0f deg");
	ImGui::DragFloat3("Scale", &object.scale.x, 0.02f, 0.05f, 50.0f);

	ImGui::SeparatorText("Material");
	if (ImGui::BeginCombo("Texture", materials[size_t(object.material)].name.c_str())) {
		for (int i = 0; i < int(materials.size()); ++i) {
			const Material& m = materials[size_t(i)];
			std::string label = m.name;
			if (m.specular.image) label += "  [specular]";
			if (m.emissive.image) label += "  [emissive]";
			if (ImGui::Selectable(label.c_str(), object.material == i)) {
				object.material = i;
			}
		}
		ImGui::EndCombo();
	}
	ImGui::ColorEdit3("Albedo tint", &object.tint.x);
	ImGui::ColorEdit3("Specular color", &object.specular.x);
	ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 512.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Emissive strength", &object.emissive_strength, 0.0f, 5.0f);
	ImGui::DragFloat2("UV scale", &object.uv_scale.x, 0.05f, 0.1f, 50.0f);
	ImGui::Checkbox("Non-trivial sampling (push constants)", &object.nontrivial_sampling);

	ImGui::SeparatorText("Animation");
	ImGui::Checkbox("Move along trajectory", &object.animate);
	if (object.animate) {
		const char* trajectories[] = { "Circle", "Lissajous figure", "Rose" };
		ImGui::Combo("Trajectory", reinterpret_cast<int*>(&object.trajectory), trajectories, 3);
		ImGui::SliderFloat("Speed", &object.speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &object.radius, 0.0f, 6.0f);
		ImGui::SliderFloat("Height", &object.height, 0.0f, 3.0f);
		ImGui::SliderFloat("Frequency A", &object.frequency_a, 1.0f, 7.0f, "%.0f");
		ImGui::SliderFloat("Frequency B", &object.frequency_b, 1.0f, 7.0f, "%.0f");
		ImGui::DragFloat3("Spin (deg/s)", &object.spin.x, 1.0f, -720.0f, 720.0f);
	}

	ImGui::PopID();
}

void drawObjectsTab() {
	if (ImGui::Button(animation_playing ? "Pause animation" : "Play animation")) {
		animation_playing = !animation_playing;
	}

	ImGui::SeparatorText("Non-trivial sampling");
	ImGui::SliderFloat("Effect speed", &effect_speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Color modulation", &effect_strength, 0.0f, 1.0f);

	ImGui::SeparatorText("Objects");
	ImGui::BeginDisabled(objects.size() >= max_objects);
	if (ImGui::Button("Add cube")) {
		addObject("", camera.position + camera.forward() * 4.0f, glm::vec3(1.0f), "Crate");
		selected_object = int(objects.size()) - 1;
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	ImGui::BeginDisabled(objects.empty());
	if (ImGui::Button("Remove selected")) {
		objects.erase(objects.begin() + selected_object);
		selected_object = std::max(0, std::min(selected_object, int(objects.size()) - 1));
	}
	ImGui::EndDisabled();

	if (ImGui::BeginListBox("##objects", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing()))) {
		for (int i = 0; i < int(objects.size()); ++i) {
			if (ImGui::Selectable(objects[size_t(i)].name.c_str(), selected_object == i)) {
				selected_object = i;
			}
		}
		ImGui::EndListBox();
	}

	if (!objects.empty()) {
		drawObjectEditor(objects[size_t(selected_object)]);
	}
}

void drawShadowsTab() {
	if (ImGui::CollapsingHeader("Sun (directional light)", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("sun_shadow");
		ImGui::Checkbox("Cast shadows", &shadows.sun_enabled);
		ImGui::SliderFloat("Azimuth", &sun.azimuth, -180.0f, 180.0f, "%.1f deg");
		ImGui::SliderFloat("Elevation", &sun.elevation, 5.0f, 90.0f, "%.1f deg");
		ImGui::SliderFloat("Ortho area (half size)", &shadows.area, 1.0f, 50.0f);
		ImGui::DragFloat3("Area center", &shadows.center.x, 0.05f);
		ImGui::SliderFloat("Sun distance", &shadows.distance, 1.0f, 100.0f);
		ImGui::DragFloatRange2("Near / far", &shadows.near_plane, &shadows.far_plane, 0.1f, 0.01f, 300.0f);
		ImGui::SliderFloat("Depth bias (constant)", &shadows.bias_constant, 0.0f, 10.0f);
		ImGui::SliderFloat("Depth bias (slope)", &shadows.bias_slope, 0.0f, 10.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Spot light shadow", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("spot_shadow");
		const std::string preview = shadows.spot_index < 0 || shadows.spot_index >= int(spot_lights.size())
			? std::string("None")
			: "Spot #" + std::to_string(shadows.spot_index);
		if (ImGui::BeginCombo("Shadow caster", preview.c_str())) {
			if (ImGui::Selectable("None", shadows.spot_index < 0)) {
				shadows.spot_index = -1;
			}
			for (int i = 0; i < int(spot_lights.size()); ++i) {
				const std::string label = "Spot #" + std::to_string(i);
				if (ImGui::Selectable(label.c_str(), shadows.spot_index == i)) {
					shadows.spot_index = i;
				}
			}
			ImGui::EndCombo();
		}
		ImGui::DragFloatRange2("Near / far", &shadows.spot_near, &shadows.spot_far, 0.05f, 0.01f, 200.0f);
		ImGui::SliderFloat("Depth bias (constant)", &shadows.spot_bias_constant, 0.0f, 10.0f);
		ImGui::SliderFloat("Depth bias (slope)", &shadows.spot_bias_slope, 0.0f, 10.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Filtering and debug", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderInt("PCF radius", &shadows.pcf_radius, 0, 2);
		ImGui::Checkbox("Show shadow maps", &shadows.show_preview);
		if (shadows.show_preview) {
			const float size = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
			ImGui::Image(ImTextureRef(ImTextureID(sun_shadow_preview)), ImVec2(size, size));
			ImGui::SameLine();
			ImGui::Image(ImTextureRef(ImTextureID(spot_shadow_preview)), ImVec2(size, size));
			ImGui::TextDisabled("Left: sun (ortho), right: spot (perspective)");
		}
	}
}

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420, 680), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab #4: Shadow mapping");

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	if (ImGui::BeginTabBar("tabs")) {
		if (ImGui::BeginTabItem("Shadows")) {
			drawShadowsTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Objects")) {
			drawObjectsTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights")) {
			drawLightsTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Camera")) {
			drawCameraTab();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	ImGui::End();
}

void updateCamera(float dt) {
	const ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw += io.MouseDelta.x * camera.mouse_sensitivity;
		camera.pitch -= io.MouseDelta.y * camera.mouse_sensitivity;
		camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
		if (camera.yaw > 180.0f) camera.yaw -= 360.0f;
		if (camera.yaw < -180.0f) camera.yaw += 360.0f;
	}

	if (io.WantCaptureKeyboard) {
		return;
	}

	const glm::vec3 forward = camera.forward();
	const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
	const glm::vec3 up(0.0f, 1.0f, 0.0f);

	glm::vec3 move(0.0f);
	if (ImGui::IsKeyDown(ImGuiKey_W)) move += forward;
	if (ImGui::IsKeyDown(ImGuiKey_S)) move -= forward;
	if (ImGui::IsKeyDown(ImGuiKey_D)) move += right;
	if (ImGui::IsKeyDown(ImGuiKey_A)) move -= right;
	if (ImGui::IsKeyDown(ImGuiKey_E)) move += up;
	if (ImGui::IsKeyDown(ImGuiKey_Q)) move -= up;

	if (glm::dot(move, move) > 0.0f) {
		const float boost = ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f;
		camera.position += glm::normalize(move) * camera.move_speed * boost * dt;
	}
}

// ---- Инициализация ----------------------------------------------------------

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }, // карта теней солнца
		{ 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }, // карта теней прожектора
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(scene_bindings)),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		return false;
	}

	// Набор материала: динамический uniform буфер моделей + кортежи изображение-сэмплер
	const VkDescriptorSetLayoutBinding material_bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
	};

	const VkDescriptorSetLayoutCreateInfo material_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(material_bindings)),
		.pBindings = material_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &material_layout_info, nullptr,
	                                &material_set_layout) != VK_SUCCESS) {
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, max_materials },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_materials * 3 + 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_materials + 1,
		.poolSizeCount = uint32_t(std::size(pool_sizes)),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		return false;
	}

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &alloc_info, &scene_descriptor_set) != VK_SUCCESS) {
		return false;
	}

	scene_uniform_buffer = vkutil::createHostBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      scene_uniform_buffer.buffer, sizeof(SceneUniforms));

	ensureStorageCapacity(point_lights_buffer, 1, sizeof(GpuPointLight), 0);
	ensureStorageCapacity(spot_lights_buffer, 2, sizeof(GpuSpotLight), 0);

	// Шаг элементов динамического буфера должен быть кратен minUniformBufferOffsetAlignment
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	model_uniform_stride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;

	model_uniform_buffer = vkutil::createHostBuffer(model_uniform_stride * max_objects,
	                                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	return true;
}

bool createTextures() {
	auto& context = graphics::internal::context;

	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = VK_LOD_CLAMP_NONE,
	};

	if (vkCreateSampler(context.device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS) {
		return false;
	}

	const uint8_t black_pixel[4] = { 0, 0, 0, 255 };
	black_texture = vkutil::createTexture(black_pixel, 1, 1, VK_FORMAT_R8G8B8A8_UNORM);

	return addMaterial("Crate", "crate_albedo.png", "crate_specular.png", nullptr) &&
	       addMaterial("Bricks", "bricks_albedo.jpg", nullptr, nullptr) &&
	       addMaterial("Tiles", "tiles_albedo.png", "tiles_specular.png", nullptr) &&
	       addMaterial("Lava stone", "lava_albedo.png", "lava_specular.png", "lava_emissive.png") &&
	       addMaterial("Sci-fi panel", "panel_albedo.png", "panel_specular.png", "panel_emissive.png") &&
	       addMaterial("Rune", "rune_albedo.png", nullptr, nullptr);
}

bool createShadowResources() {
	auto& context = graphics::internal::context;

	const VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	sun_shadow_map = vkutil::createImage(sun_shadow_map_size, sun_shadow_map_size, shadow_map_format,
	                                     usage, VK_IMAGE_ASPECT_DEPTH_BIT);
	spot_shadow_map = vkutil::createImage(spot_shadow_map_size, spot_shadow_map_size, shadow_map_format,
	                                      usage, VK_IMAGE_ASPECT_DEPTH_BIT);
	if (!sun_shadow_map.image || !spot_shadow_map.image) {
		return false;
	}

	// Сэмплер со сравнением: результат выборки — 1, если глубина фрагмента
	// меньше глубины в карте (точка освещена), и 0 иначе. Линейная фильтрация
	// даёт аппаратное сглаживание 2x2. За пределами карты — «освещено».
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.maxLod = 0.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
	};

	if (vkCreateSampler(context.device, &sampler_info, nullptr, &shadow_sampler) != VK_SUCCESS) {
		return false;
	}

	// Перевод карт в SHADER_READ_ONLY, чтобы они были в корректном состоянии до первого кадра
	vkutil::immediateSubmit([](VkCommandBuffer cmd) {
		for (VkImage image : { sun_shadow_map.image, spot_shadow_map.image }) {
			const VkImageMemoryBarrier barrier = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = image,
				.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
			};
			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			                     0, 0, nullptr, 0, nullptr, 1, &barrier);
		}
	});

	// Кортежи «карта теней + сэмплер сравнения» в наборе дескрипторов сцены
	writeImageDescriptor(scene_descriptor_set, 3, sun_shadow_map.view, shadow_sampler);
	writeImageDescriptor(scene_descriptor_set, 4, spot_shadow_map.view, shadow_sampler);

	sun_shadow_preview = ImGui_ImplVulkan_AddTexture(sun_shadow_map.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	spot_shadow_preview = ImGui_ImplVulkan_AddTexture(spot_shadow_map.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	return true;
}

bool createPipelines() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayout lit_layouts[] = { scene_set_layout, material_set_layout };

	const VkPushConstantRange push_constants = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(EffectPushConstants),
	};

	const VkPipelineLayoutCreateInfo lit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constants,
	};

	if (vkCreatePipelineLayout(context.device, &lit_layout_info, nullptr, &lit_pipeline_layout) != VK_SUCCESS) {
		return false;
	}

	const VkPipelineLayoutCreateInfo unlit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &unlit_layout_info, nullptr, &unlit_pipeline_layout) != VK_SUCCESS) {
		return false;
	}

	const VkPushConstantRange shadow_push_constants = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(ShadowPushConstants),
	};

	const VkPipelineLayoutCreateInfo shadow_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &shadow_push_constants,
	};

	if (vkCreatePipelineLayout(context.device, &shadow_layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
		return false;
	}

	VkShaderModule lit_vert = vkutil::loadShaderModule(LAB_SHADER_DIR "lit.vert.spv");
	VkShaderModule lit_frag = vkutil::loadShaderModule(LAB_SHADER_DIR "lit.frag.spv");
	VkShaderModule unlit_vert = vkutil::loadShaderModule(LAB_SHADER_DIR "unlit.vert.spv");
	VkShaderModule unlit_frag = vkutil::loadShaderModule(LAB_SHADER_DIR "unlit.frag.spv");
	VkShaderModule shadow_vert = vkutil::loadShaderModule(LAB_SHADER_DIR "shadow.vert.spv");
	if (!lit_vert || !lit_frag || !unlit_vert || !unlit_frag || !shadow_vert) {
		return false;
	}

	vkutil::PipelineConfig lit;
	lit.vertex_shader = lit_vert;
	lit.fragment_shader = lit_frag;
	lit.bindings = { { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };
	lit.attributes = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
		{ 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
	};
	lit.layout = lit_pipeline_layout;
	lit.render_pass = context.render_pass;
	lit_pipeline = vkutil::createGraphicsPipeline(lit);

	vkutil::PipelineConfig unlit;
	unlit.vertex_shader = unlit_vert;
	unlit.fragment_shader = unlit_frag;
	unlit.bindings = { { 0, sizeof(DebugVertex), VK_VERTEX_INPUT_RATE_VERTEX } };
	unlit.attributes = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(DebugVertex, position) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(DebugVertex, color) },
	};
	unlit.layout = unlit_pipeline_layout;
	unlit.render_pass = context.render_pass;
	unlit_triangles_pipeline = vkutil::createGraphicsPipeline(unlit);

	unlit.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	unlit.cull_mode = VK_CULL_MODE_NONE;
	unlit_lines_pipeline = vkutil::createGraphicsPipeline(unlit);

	// Конвейер карты теней: без render pass (динамический рендеринг), без цветовых
	// вложений и без фрагментного шейдера; сдвиг глубины задаётся динамически.
	// Отсечение граней выключено, чтобы тонкие объекты тоже отбрасывали тень.
	vkutil::PipelineConfig shadow;
	shadow.vertex_shader = shadow_vert;
	shadow.bindings = { { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };
	shadow.attributes = { { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) } };
	shadow.cull_mode = VK_CULL_MODE_NONE;
	shadow.depth_compare = VK_COMPARE_OP_LESS;
	shadow.dynamic_depth_bias = true;
	shadow.layout = shadow_pipeline_layout;
	shadow.depth_format = shadow_map_format;
	shadow_pipeline = vkutil::createGraphicsPipeline(shadow);

	vkDestroyShaderModule(context.device, lit_vert, nullptr);
	vkDestroyShaderModule(context.device, lit_frag, nullptr);
	vkDestroyShaderModule(context.device, unlit_vert, nullptr);
	vkDestroyShaderModule(context.device, unlit_frag, nullptr);
	vkDestroyShaderModule(context.device, shadow_vert, nullptr);

	return lit_pipeline && unlit_triangles_pipeline && unlit_lines_pipeline && shadow_pipeline;
}

void createScene() {
	Object* o = &addObject("Floor", glm::vec3(0.0f, -0.6f, 0.0f), glm::vec3(24.0f, 0.2f, 24.0f), "Tiles");
	o->uv_scale = glm::vec2(6.0f);
	o->shininess = 64.0f;

	o = &addObject("Back wall", glm::vec3(0.0f, 3.5f, -8.0f), glm::vec3(24.0f, 8.0f, 0.3f), "Bricks");
	o->uv_scale = glm::vec2(6.0f, 2.0f);
	o->specular = glm::vec3(0.1f);

	o = &addObject("Magic cube (animated)", glm::vec3(0.0f, 1.2f, 0.0f), glm::vec3(1.0f), "Rune");
	o->animate = true;
	o->speed = 0.4f;
	o->nontrivial_sampling = true;
	o->shininess = 64.0f;

	o = &addObject("Crate", glm::vec3(-4.0f, 0.5f, -2.0f), glm::vec3(2.0f), "Crate");
	o->rotation = glm::vec3(0.0f, 30.0f, 0.0f);
	o->shininess = 96.0f;

	o = &addObject("Lava pillar", glm::vec3(4.0f, 1.5f, -3.0f), glm::vec3(1.0f, 4.0f, 1.0f), "Lava stone");
	o->uv_scale = glm::vec2(1.0f, 4.0f);
	o->emissive_strength = 2.0f;

	o = &addObject("Sci-fi box", glm::vec3(3.0f, 0.0f, 2.5f), glm::vec3(1.0f), "Sci-fi panel");
	o->rotation = glm::vec3(0.0f, 45.0f, 0.0f);
	o->shininess = 128.0f;
	o->emissive_strength = 1.5f;

	o = &addObject("Small crate", glm::vec3(-2.2f, 0.0f, 0.2f), glm::vec3(1.0f), "Crate");
	o->rotation = glm::vec3(0.0f, -15.0f, 0.0f);

	// Арка — чтобы тень была интереснее
	o = &addObject("Arch left", glm::vec3(-1.5f, 1.5f, -5.5f), glm::vec3(0.6f, 4.0f, 0.6f), "Bricks");
	o->uv_scale = glm::vec2(0.5f, 2.0f);
	o = &addObject("Arch right", glm::vec3(1.5f, 1.5f, -5.5f), glm::vec3(0.6f, 4.0f, 0.6f), "Bricks");
	o->uv_scale = glm::vec2(0.5f, 2.0f);
	o = &addObject("Arch top", glm::vec3(0.0f, 3.8f, -5.5f), glm::vec3(3.6f, 0.6f, 0.6f), "Bricks");
	o->uv_scale = glm::vec2(3.0f, 0.5f);

	PointLight p;
	p.position = glm::vec3(0.0f, 2.5f, 0.0f);
	p.color = glm::vec3(1.0f, 0.6f, 0.3f);
	p.intensity = 8.0f;
	p.orbit = true;
	p.orbit_radius = 4.0f;
	point_lights.push_back(p);

	p.position = glm::vec3(-5.0f, 1.0f, 3.0f);
	p.color = glm::vec3(0.3f, 0.6f, 1.0f);
	p.intensity = 5.0f;
	p.orbit = false;
	point_lights.push_back(p);

	SpotLight s;
	s.position = glm::vec3(2.0f, 4.0f, 2.0f);
	s.yaw = -66.6f;
	s.pitch = -41.3f;
	s.inner_angle = 20.0f;
	s.outer_angle = 30.0f;
	s.color = glm::vec3(1.0f, 1.0f, 0.9f);
	s.intensity = 40.0f;
	spot_lights.push_back(s);
}

// Рисует всю сцену в карту теней с точки зрения источника.
// Используется динамический рендеринг: вложение глубины указывается прямо
// в vkCmdBeginRendering, без VkRenderPass и VkFramebuffer.
void recordShadowPass(VkCommandBuffer cmd, const vkutil::Image& shadow_map,
                      const glm::mat4& light_view_projection, float bias_constant, float bias_slope) {
	// SHADER_READ_ONLY (чтение в прошлом кадре) -> вложение глубины
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, // старое содержимое не нужно
		.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = shadow_map.image,
		.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 },
	};
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);

	const VkRenderingAttachmentInfo depth_attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = shadow_map.view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderingInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .extent = { shadow_map.width, shadow_map.height } },
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pDepthAttachment = &depth_attachment,
	};

	vkCmdBeginRendering(cmd, &rendering_info);

	const VkViewport viewport = {
		.width = float(shadow_map.width),
		.height = float(shadow_map.height),
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = { shadow_map.width, shadow_map.height } };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);
	vkCmdSetDepthBias(cmd, bias_constant, 0.0f, bias_slope);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
	                        0, 1, &scene_descriptor_set, 0, nullptr);

	const ShadowPushConstants push = { light_view_projection };
	vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (size_t i = 0; i < objects.size(); ++i) {
		const uint32_t dynamic_offset = uint32_t(i * model_uniform_stride);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
		                        1, 1, &materials[size_t(objects[i].material)].descriptor_set,
		                        1, &dynamic_offset);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	// Вложение глубины -> текстура для чтения во фрагментном шейдере
	barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

bool initialize() {
	if (!createDescriptors() || !createTextures() || !createShadowResources() || !createPipelines()) {
		return false;
	}

	createCubeGeometry();
	debug_vertex_buffer = vkutil::createHostBuffer(sizeof(DebugVertex) * 4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	createScene();

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	for (auto& material : materials) {
		vkutil::destroyImage(material.albedo);
		vkutil::destroyImage(material.specular);
		vkutil::destroyImage(material.emissive);
	}
	materials.clear();
	vkutil::destroyImage(black_texture);
	ImGui_ImplVulkan_RemoveTexture(sun_shadow_preview);
	ImGui_ImplVulkan_RemoveTexture(spot_shadow_preview);
	vkutil::destroyImage(sun_shadow_map);
	vkutil::destroyImage(spot_shadow_map);
	vkDestroySampler(context.device, shadow_sampler, nullptr);
	vkDestroySampler(context.device, texture_sampler, nullptr);

	vkutil::destroyBuffer(vertex_buffer);
	vkutil::destroyBuffer(index_buffer);
	vkutil::destroyBuffer(scene_uniform_buffer);
	vkutil::destroyBuffer(model_uniform_buffer);
	vkutil::destroyBuffer(point_lights_buffer);
	vkutil::destroyBuffer(spot_lights_buffer);
	vkutil::destroyBuffer(debug_vertex_buffer);

	vkDestroyPipeline(context.device, lit_pipeline, nullptr);
	vkDestroyPipeline(context.device, unlit_triangles_pipeline, nullptr);
	vkDestroyPipeline(context.device, unlit_lines_pipeline, nullptr);
	vkDestroyPipeline(context.device, shadow_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, shadow_pipeline_layout, nullptr);
	vkDestroyPipelineLayout(context.device, lit_pipeline_layout, nullptr);
	vkDestroyPipelineLayout(context.device, unlit_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);

	vkutil::shutdown();
}

void update(double time) {
	const float dt = float(time - previous_time);
	previous_time = time;

	updateCamera(dt);

	if (animation_playing) {
		effect_time += dt;
		for (auto& object : objects) {
			if (object.animate) {
				object.time += dt * object.speed;
			}
		}
		for (auto& light : point_lights) {
			if (light.orbit) {
				light.orbit_phase += dt * light.orbit_speed;
			}
		}
	}

	drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	// ---- Обновление буферов (предыдущий кадр на GPU уже завершён) ----

	const int spot_shadow_index = shadows.spot_index < int(spot_lights.size()) ? shadows.spot_index : -1;
	const glm::mat4 sun_view_projection = sunViewProjection();
	const glm::mat4 spot_view_projection = spot_shadow_index >= 0
		? spotViewProjection(spot_lights[size_t(spot_shadow_index)])
		: glm::mat4(1.0f);

	const SceneUniforms scene = {
		.view_projection = viewProjectionMatrix(),
		.camera_position = glm::vec4(camera.position, 1.0f),
		.ambient = glm::vec4(ambient_color * ambient_intensity, 0.0f),
		.sun_direction = glm::vec4(sun.direction(), 0.0f),
		.sun_color = glm::vec4(sun.color * sun.intensity, 0.0f),
		.counts = glm::uvec4(point_lights.size(), spot_lights.size(), 0, 0),
		.sun_view_projection = sun_view_projection,
		.spot_view_projection = spot_view_projection,
		.shadow_params = glm::vec4(float(shadows.pcf_radius), shadows.sun_enabled ? 1.0f : 0.0f,
		                           float(spot_shadow_index), 0.0f),
		.shadow_texel = glm::vec4(glm::vec2(1.0f / sun_shadow_map_size), glm::vec2(1.0f / spot_shadow_map_size)),
	};
	std::memcpy(scene_uniform_buffer.mapped, &scene, sizeof(scene));

	ensureStorageCapacity(point_lights_buffer, 1, sizeof(GpuPointLight), point_lights.size());
	ensureStorageCapacity(spot_lights_buffer, 2, sizeof(GpuSpotLight), spot_lights.size());

	auto* gpu_points = static_cast<GpuPointLight*>(point_lights_buffer.mapped);
	for (size_t i = 0; i < point_lights.size(); ++i) {
		gpu_points[i] = {
			.position = glm::vec4(point_lights[i].currentPosition(), 1.0f),
			.color = glm::vec4(point_lights[i].color * point_lights[i].intensity, 0.0f),
		};
	}

	auto* gpu_spots = static_cast<GpuSpotLight*>(spot_lights_buffer.mapped);
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLight& light = spot_lights[i];
		gpu_spots[i] = {
			.position = glm::vec4(light.position, 1.0f),
			.direction = glm::vec4(light.direction(), 0.0f),
			.color = glm::vec4(light.color * light.intensity, 0.0f),
			.angles = glm::vec4(std::cos(glm::radians(light.inner_angle)),
			                    std::cos(glm::radians(light.outer_angle)), 0.0f, 0.0f),
		};
	}

	// Свойства всех моделей записываются в один динамический uniform буфер
	auto* model_data = static_cast<uint8_t*>(model_uniform_buffer.mapped);
	for (size_t i = 0; i < objects.size(); ++i) {
		const Object& object = objects[i];
		const glm::mat4 model = modelMatrix(object);
		const ModelUniforms uniforms = {
			.model = model,
			.normal_matrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(model)))),
			.tint = glm::vec4(object.tint, 1.0f),
			.specular = glm::vec4(object.specular, object.shininess),
			.uv_transform = glm::vec4(object.uv_scale, 0.0f, 0.0f),
			.params = glm::vec4(object.emissive_strength, 0.0f, 0.0f, 0.0f),
		};
		std::memcpy(model_data + i * model_uniform_stride, &uniforms, sizeof(uniforms));
	}

	std::vector<DebugVertex> debug_triangles;
	std::vector<DebugVertex> debug_lines;
	if (show_light_markers) {
		for (const auto& light : point_lights) {
			appendMarkerCube(debug_triangles, light.currentPosition(), 0.12f, light.color);
		}
		for (const auto& light : spot_lights) {
			appendMarkerCube(debug_triangles, light.position, 0.12f, light.color);
			appendSpotCone(debug_lines, light);
		}
	}

	const size_t debug_vertex_count = debug_triangles.size() + debug_lines.size();
	if (sizeof(DebugVertex) * debug_vertex_count > debug_vertex_buffer.size) {
		vkutil::destroyBuffer(debug_vertex_buffer);
		debug_vertex_buffer = vkutil::createHostBuffer(sizeof(DebugVertex) * debug_vertex_count * 2,
		                                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	}
	auto* debug_vertices = static_cast<DebugVertex*>(debug_vertex_buffer.mapped);
	std::copy(debug_triangles.begin(), debug_triangles.end(), debug_vertices);
	std::copy(debug_lines.begin(), debug_lines.end(), debug_vertices + debug_triangles.size());

	// ---- Запись команд ----

	VkCommandBuffer cmd = fd.command_buffer;

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	// ---- Проходы теней (динамический рендеринг) ----
	if (shadows.sun_enabled) {
		recordShadowPass(cmd, sun_shadow_map, sun_view_projection, shadows.bias_constant, shadows.bias_slope);
	}
	if (spot_shadow_index >= 0) {
		recordShadowPass(cmd, spot_shadow_map, spot_view_projection,
		                 shadows.spot_bias_constant, shadows.spot_bias_slope);
	}

	// ---- Основной проход ----
	const VkClearValue clear_values[] = {
		{ .color = { { clear_color.r, clear_color.g, clear_color.b, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout,
	                        0, 1, &scene_descriptor_set, 0, nullptr);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (size_t i = 0; i < objects.size(); ++i) {
		const Object& object = objects[i];

		// Набор дескрипторов выбранного материала + сдвиг до свойств модели
		const uint32_t dynamic_offset = uint32_t(i * model_uniform_stride);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout,
		                        1, 1, &materials[size_t(object.material)].descriptor_set,
		                        1, &dynamic_offset);

		const EffectPushConstants effect = {
			.flags = object.nontrivial_sampling ? effect_enabled : 0u,
			.time = effect_time,
			.speed = effect_speed,
			.strength = effect_strength,
		};
		vkCmdPushConstants(cmd, lit_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(effect), &effect);

		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	if (debug_vertex_count > 0) {
		vkCmdBindVertexBuffers(cmd, 0, 1, &debug_vertex_buffer.buffer, &offset);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, unlit_triangles_pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, unlit_pipeline_layout,
		                        0, 1, &scene_descriptor_set, 0, nullptr);
		vkCmdDraw(cmd, uint32_t(debug_triangles.size()), 1, 0, 0);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, unlit_lines_pipeline);
		vkCmdDraw(cmd, uint32_t(debug_lines.size()), 1, uint32_t(debug_triangles.size()), 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
