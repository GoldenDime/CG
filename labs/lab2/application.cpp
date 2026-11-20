// =============================================================================
//  Лабораторная работа №2: Камера и освещение
//  (продолжение ЛР №1, вариант 1: куб)
//
//  Базовое задание: свободная камера, направленный источник света, рассеянный
//  свет и материал по модели Блинн-Фонга, всё настраивается из ImGui.
//  У вершин куба есть процедурно сгенерированные нормали.
//  Дополнительные задания:
//   1. Два набора дескрипторов: набор сцены (uniform буфер + storage буферы
//      источников) и набор модели (матрица трансформации + материал)
//   2. Массив точечных источников в storage буфере, затухание по закону
//      обратных квадратов, добавление/удаление из интерфейса
//   3. Массив прожекторных источников в storage буфере, затухание и мягкие
//      края (внутренний и внешний углы), добавление/удаление из интерфейса
//   4. Визуализация источников: яркие кубы цвета источника, для прожекторов —
//      конус белыми линиями. Не освещаются, можно скрыть.
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

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vk_utils.hpp"

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

struct DebugVertex {
	glm::vec3 position;
	glm::vec3 color;
};

// ---- Структуры, разделяемые с шейдерами (std140 / std430) -------------------

struct SceneUniforms {
	glm::mat4 view_projection;
	glm::vec4 camera_position;
	glm::vec4 ambient;
	glm::vec4 sun_direction;
	glm::vec4 sun_color;
	glm::uvec4 counts;
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

struct ModelUniforms {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 albedo;
	glm::vec4 specular;
};

// ---- Сцена ------------------------------------------------------------------

enum class Trajectory : int {
	Circle,
	Lissajous,
	Rose,
};

struct Material {
	glm::vec3 albedo{0.8f};
	glm::vec3 specular{0.5f};
	float shininess = 32.0f;
};

struct Object {
	std::string name;

	glm::vec3 position{0.0f};
	glm::vec3 rotation{0.0f};
	glm::vec3 scale{1.0f};

	Material material;

	// Анимация из ЛР №1
	bool animate = false;
	Trajectory trajectory = Trajectory::Lissajous;
	float speed = 1.0f;
	float radius = 2.0f;
	float height = 0.5f;
	float frequency_a = 3.0f;
	float frequency_b = 2.0f;
	glm::vec3 spin{30.0f, 90.0f, 0.0f};
	float time = 0.0f;

	vkutil::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct Camera {
	glm::vec3 position{0.0f, 3.0f, 9.0f};
	float yaw = 0.0f;      // градусы, 0 — взгляд вдоль -Z
	float pitch = -15.0f;  // градусы
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
	float azimuth = 35.0f;    // градусы
	float elevation = 50.0f;  // градусы
	glm::vec3 color{1.0f, 0.95f, 0.85f};
	float intensity = 0.6f;

	glm::vec3 direction() const { // направление НА источник
		const float a = glm::radians(azimuth);
		const float e = glm::radians(elevation);
		return { std::cos(e) * std::sin(a), std::sin(e), std::cos(e) * std::cos(a) };
	}
};

struct PointLight {
	glm::vec3 position{0.0f, 2.0f, 0.0f};
	glm::vec3 color{1.0f};
	float intensity = 8.0f;

	// Движение по окружности вокруг position
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
	float yaw = 0.0f;       // направление луча
	float pitch = -90.0f;
	glm::vec3 color{1.0f};
	float intensity = 30.0f;
	float inner_angle = 15.0f; // градусы, от оси до края
	float outer_angle = 25.0f;

	glm::vec3 direction() const {
		const float y = glm::radians(yaw);
		const float p = glm::radians(pitch);
		return { std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p) };
	}
};

// ---- Ресурсы Vulkan ---------------------------------------------------------

VkDescriptorSetLayout scene_set_layout;
VkDescriptorSetLayout model_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_descriptor_set;

VkPipelineLayout lit_pipeline_layout;
VkPipelineLayout unlit_pipeline_layout;
VkPipeline lit_pipeline;
VkPipeline unlit_triangles_pipeline;
VkPipeline unlit_lines_pipeline;

vkutil::Buffer vertex_buffer;
vkutil::Buffer index_buffer;
uint32_t index_count;

vkutil::Buffer scene_uniform_buffer;
vkutil::Buffer point_lights_buffer;
vkutil::Buffer spot_lights_buffer;
vkutil::Buffer debug_vertex_buffer;

// ---- Состояние --------------------------------------------------------------

std::vector<Object> objects;
int selected_object = 0;
int created_objects = 0;

Camera camera;

glm::vec3 ambient_color{0.6f, 0.7f, 1.0f};
float ambient_intensity = 0.08f;
DirectionalLight sun;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;
int selected_point_light = 0;
int selected_spot_light = 0;

bool show_light_markers = true;
bool animation_playing = true;
double previous_time = 0.0;

glm::vec3 clear_color{0.02f, 0.02f, 0.03f};

// ---- Геометрия --------------------------------------------------------------

// Куб с ребром 1: 6 граней по 4 вершины (у каждой грани своя нормаль).
// Нормаль грани n, а векторы u и v лежат в плоскости грани, u x v = n,
// поэтому обход (-u,-v) -> (+u,-v) -> (+u,+v) -> (-u,+v) идёт против часовой стрелки.
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
			vertices.push_back({ 0.5f * (face.n + c.x * face.u + c.y * face.v), face.n });
		}
		indices.insert(indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
	}

	index_count = uint32_t(indices.size());

	vertex_buffer = vkutil::createHostBuffer(sizeof(Vertex) * vertices.size(),
	                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	std::memcpy(vertex_buffer.mapped, vertices.data(), sizeof(Vertex) * vertices.size());

	index_buffer = vkutil::createHostBuffer(sizeof(uint32_t) * indices.size(),
	                                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	std::memcpy(index_buffer.mapped, indices.data(), sizeof(uint32_t) * indices.size());
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

// Увеличивает storage буфер, если источников стало больше, чем помещается,
// и перепривязывает его к набору дескрипторов сцены.
// Вызывается из render(), когда предыдущий кадр уже завершён на GPU.
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

void addObject(const std::string& name, const glm::vec3& position, const glm::vec3& scale,
               const Material& material) {
	auto& context = graphics::internal::context;

	Object object;
	object.name = name.empty() ? "Cube #" + std::to_string(++created_objects) : name;
	object.position = position;
	object.scale = scale;
	object.material = material;
	object.uniform_buffer = vkutil::createHostBuffer(sizeof(ModelUniforms),
	                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &model_set_layout,
	};
	vkAllocateDescriptorSets(context.device, &alloc_info, &object.descriptor_set);

	writeBufferDescriptor(object.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      object.uniform_buffer.buffer, sizeof(ModelUniforms));

	objects.push_back(std::move(object));
}

void removeObject(size_t index) {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeDescriptorSets(context.device, descriptor_pool, 1, &objects[index].descriptor_set);
	vkutil::destroyBuffer(objects[index].uniform_buffer);
	objects.erase(objects.begin() + ptrdiff_t(index));
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

	// Ось прожектора
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

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		PointLight prototype;
		prototype.position = camera.position + camera.forward() * 3.0f;
		drawLightList("Point", point_lights, selected_point_light, prototype, drawPointLightEditor);
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
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

	ImGui::SeparatorText("Material (Blinn-Phong)");
	ImGui::ColorEdit3("Albedo", &object.material.albedo.x);
	ImGui::ColorEdit3("Specular color", &object.material.specular.x);
	ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.0f", ImGuiSliderFlags_Logarithmic);

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

	if (ImGui::Button("Add cube")) {
		addObject("", camera.position + camera.forward() * 4.0f, glm::vec3(1.0f), Material{});
		selected_object = int(objects.size()) - 1;
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(objects.empty());
	if (ImGui::Button("Remove selected")) {
		removeObject(size_t(selected_object));
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

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400, 680), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab #2: Camera and lighting");

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	if (ImGui::BeginTabBar("tabs")) {
		if (ImGui::BeginTabItem("Lights")) {
			drawLightsTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Objects")) {
			drawObjectsTab();
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

	// Набор сцены: uniform буфер + два storage буфера с источниками
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
		{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
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

	// Набор модели: uniform буфер с матрицей и материалом
	const VkDescriptorSetLayoutBinding model_binding = {
		0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		return false;
	}

	constexpr uint32_t max_objects = 128;

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, max_objects + 1 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = max_objects + 1,
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

	return true;
}

bool createPipelines() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayout lit_layouts[] = { scene_set_layout, model_set_layout };

	const VkPipelineLayoutCreateInfo lit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_layouts,
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

	VkShaderModule lit_vert = vkutil::loadShaderModule(LAB_SHADER_DIR "lit.vert.spv");
	VkShaderModule lit_frag = vkutil::loadShaderModule(LAB_SHADER_DIR "lit.frag.spv");
	VkShaderModule unlit_vert = vkutil::loadShaderModule(LAB_SHADER_DIR "unlit.vert.spv");
	VkShaderModule unlit_frag = vkutil::loadShaderModule(LAB_SHADER_DIR "unlit.frag.spv");
	if (!lit_vert || !lit_frag || !unlit_vert || !unlit_frag) {
		return false;
	}

	vkutil::PipelineConfig lit;
	lit.vertex_shader = lit_vert;
	lit.fragment_shader = lit_frag;
	lit.bindings = { { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };
	lit.attributes = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
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

	vkDestroyShaderModule(context.device, lit_vert, nullptr);
	vkDestroyShaderModule(context.device, lit_frag, nullptr);
	vkDestroyShaderModule(context.device, unlit_vert, nullptr);
	vkDestroyShaderModule(context.device, unlit_frag, nullptr);

	return lit_pipeline && unlit_triangles_pipeline && unlit_lines_pipeline;
}

void createScene() {
	addObject("Floor", glm::vec3(0.0f, -0.6f, 0.0f), glm::vec3(24.0f, 0.2f, 24.0f),
	          { glm::vec3(0.55f, 0.55f, 0.6f), glm::vec3(0.2f), 16.0f });
	addObject("Back wall", glm::vec3(0.0f, 3.5f, -8.0f), glm::vec3(24.0f, 8.0f, 0.3f),
	          { glm::vec3(0.6f, 0.45f, 0.4f), glm::vec3(0.05f), 8.0f });

	addObject("Cube (animated)", glm::vec3(0.0f, 1.2f, 0.0f), glm::vec3(1.0f),
	          { glm::vec3(0.9f, 0.9f, 0.9f), glm::vec3(1.0f), 64.0f });
	objects.back().animate = true;
	objects.back().speed = 0.4f;

	addObject("Red cube", glm::vec3(-4.0f, 0.5f, -2.0f), glm::vec3(2.0f),
	          { glm::vec3(0.8f, 0.15f, 0.1f), glm::vec3(0.8f), 128.0f });
	objects.back().rotation = glm::vec3(0.0f, 30.0f, 0.0f);

	addObject("Green pillar", glm::vec3(4.0f, 1.5f, -3.0f), glm::vec3(1.0f, 4.0f, 1.0f),
	          { glm::vec3(0.2f, 0.7f, 0.3f), glm::vec3(0.3f), 32.0f });

	addObject("Blue cube", glm::vec3(3.0f, 0.0f, 2.5f), glm::vec3(1.0f),
	          { glm::vec3(0.2f, 0.35f, 0.9f), glm::vec3(1.0f), 256.0f });
	objects.back().rotation = glm::vec3(0.0f, 45.0f, 0.0f);

	PointLight p;
	p.position = glm::vec3(0.0f, 2.5f, 0.0f);
	p.color = glm::vec3(1.0f, 0.5f, 0.2f);
	p.intensity = 10.0f;
	p.orbit = true;
	p.orbit_radius = 4.0f;
	point_lights.push_back(p);

	p.position = glm::vec3(-5.0f, 1.0f, 3.0f);
	p.color = glm::vec3(0.3f, 0.6f, 1.0f);
	p.intensity = 6.0f;
	p.orbit = false;
	point_lights.push_back(p);

	SpotLight s;
	s.position = glm::vec3(-4.0f, 5.0f, 2.0f);
	s.yaw = 0.0f;
	s.pitch = -70.0f;
	s.color = glm::vec3(1.0f, 1.0f, 0.9f);
	s.intensity = 40.0f;
	spot_lights.push_back(s);

	s.position = glm::vec3(4.0f, 6.0f, 3.0f);
	s.yaw = -30.0f;
	s.pitch = -60.0f;
	s.color = glm::vec3(0.6f, 1.0f, 0.6f);
	s.inner_angle = 10.0f;
	s.outer_angle = 30.0f;
	spot_lights.push_back(s);
}

} // namespace

bool initialize() {
	if (!createDescriptors() || !createPipelines()) {
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

	for (auto& object : objects) {
		vkutil::destroyBuffer(object.uniform_buffer);
	}
	objects.clear();

	vkutil::destroyBuffer(vertex_buffer);
	vkutil::destroyBuffer(index_buffer);
	vkutil::destroyBuffer(scene_uniform_buffer);
	vkutil::destroyBuffer(point_lights_buffer);
	vkutil::destroyBuffer(spot_lights_buffer);
	vkutil::destroyBuffer(debug_vertex_buffer);

	vkDestroyPipeline(context.device, lit_pipeline, nullptr);
	vkDestroyPipeline(context.device, unlit_triangles_pipeline, nullptr);
	vkDestroyPipeline(context.device, unlit_lines_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, lit_pipeline_layout, nullptr);
	vkDestroyPipelineLayout(context.device, unlit_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);

	vkutil::shutdown();
}

void update(double time) {
	const float dt = float(time - previous_time);
	previous_time = time;

	updateCamera(dt);

	if (animation_playing) {
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

	const SceneUniforms scene = {
		.view_projection = viewProjectionMatrix(),
		.camera_position = glm::vec4(camera.position, 1.0f),
		.ambient = glm::vec4(ambient_color * ambient_intensity, 0.0f),
		.sun_direction = glm::vec4(sun.direction(), 0.0f),
		.sun_color = glm::vec4(sun.color * sun.intensity, 0.0f),
		.counts = glm::uvec4(point_lights.size(), spot_lights.size(), 0, 0),
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

	for (auto& object : objects) {
		const glm::mat4 model = modelMatrix(object);
		const ModelUniforms uniforms = {
			.model = model,
			.normal_matrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(model)))),
			.albedo = glm::vec4(object.material.albedo, 1.0f),
			.specular = glm::vec4(object.material.specular, object.material.shininess),
		};
		std::memcpy(object.uniform_buffer.mapped, &uniforms, sizeof(uniforms));
	}

	// Геометрия визуализации источников: сначала треугольники, затем линии
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

	// Освещённые объекты: набор сцены привязывается один раз, набор модели — для каждого объекта
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout,
	                        0, 1, &scene_descriptor_set, 0, nullptr);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	for (const auto& object : objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, lit_pipeline_layout,
		                        1, 1, &object.descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	// Визуализация источников света
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
