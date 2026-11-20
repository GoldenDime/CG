// =============================================================================
//  Лабораторная работа №1: Основы 3D-графики
//  Вариант 1: куб
//
//  Базовое задание: куб, спроецированный на экран.
//  Дополнительные задания:
//   1. Переключение матрицы проекции (перспективная / ортографическая)
//   2. Изменение позиции, поворота и масштаба фигуры из интерфейса
//   3. Движение и вращение по сложной траектории, пауза/воспроизведение,
//      скорость и параметры траектории
//   4. Изменение цвета фигуры (ImGui::ColorEdit)
//   5. Процедурные цвета вершин (от локальной позиции), цвет из интерфейса
//      умножается на цвет вершины
//   6. Несколько объектов на сцене, у каждого свой VkDescriptorSet
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
	glm::vec3 color;
};

// Должна совпадать с блоком ObjectUniforms в cube.vert (std140)
struct ObjectUniforms {
	glm::mat4 mvp;
	glm::vec4 color;
	glm::vec4 params;
};

enum class Trajectory : int {
	Circle,
	Lissajous,
	Rose,
};

struct Object {
	std::string name;

	// Трансформации, задаваемые из интерфейса
	glm::vec3 position{0.0f};
	glm::vec3 rotation{0.0f}; // углы Эйлера, градусы
	glm::vec3 scale{1.0f};

	glm::vec3 color{1.0f};
	bool vertex_colors = true;

	// Анимация движения по траектории
	bool animate = false;
	Trajectory trajectory = Trajectory::Lissajous;
	float speed = 1.0f;          // скорость движения по траектории
	float radius = 2.0f;         // радиус траектории
	float height = 0.75f;        // амплитуда по вертикали
	float frequency_a = 3.0f;    // частоты для фигуры Лиссажу / число лепестков розы
	float frequency_b = 2.0f;
	glm::vec3 spin{30.0f, 90.0f, 0.0f}; // скорость вращения, градусов за единицу времени
	float time = 0.0f;           // локальное время анимации

	vkutil::Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

enum class Projection : int {
	Perspective,
	Orthographic,
};

VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

vkutil::Buffer vertex_buffer;
vkutil::Buffer index_buffer;
uint32_t index_count;

std::vector<Object> objects;
int selected_object = 0;
int created_objects = 0;

Projection projection = Projection::Perspective;
float fov_degrees = 60.0f;
float ortho_height = 6.0f;
float near_plane = 0.1f;
float far_plane = 100.0f;

glm::vec3 camera_position{0.0f, 3.0f, 8.0f};
glm::vec3 camera_target{0.0f, 0.0f, 0.0f};

bool animation_playing = true;
double previous_time = 0.0;

glm::vec3 clear_color{0.08f, 0.09f, 0.12f};

// Куб с ребром 1, центр в начале координат: 8 вершин, 12 треугольников.
// Цвет вершины задаётся процедурно из её позиции в локальной системе координат:
// (x, y, z) из [-0.5; 0.5] переводится в RGB из [0; 1].
void createCubeGeometry() {
	std::vector<Vertex> vertices;
	for (int i = 0; i < 8; ++i) {
		const glm::vec3 p = {
			(i & 1) ? 0.5f : -0.5f,
			(i & 2) ? 0.5f : -0.5f,
			(i & 4) ? 0.5f : -0.5f,
		};
		vertices.push_back({ p, p + glm::vec3(0.5f) });
	}

	// Обход против часовой стрелки, если смотреть на грань снаружи
	const uint32_t indices[] = {
		0, 4, 6,  0, 6, 2, // -X
		1, 3, 7,  1, 7, 5, // +X
		0, 1, 5,  0, 5, 4, // -Y
		2, 6, 7,  2, 7, 3, // +Y
		0, 2, 3,  0, 3, 1, // -Z
		4, 5, 7,  4, 7, 6, // +Z
	};
	index_count = uint32_t(std::size(indices));

	vertex_buffer = vkutil::createHostBuffer(sizeof(Vertex) * vertices.size(),
	                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	std::memcpy(vertex_buffer.mapped, vertices.data(), sizeof(Vertex) * vertices.size());

	index_buffer = vkutil::createHostBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	std::memcpy(index_buffer.mapped, indices, sizeof(indices));
}

// Каждому объекту выделяется собственный uniform буфер и набор дескрипторов,
// указывающий на этот буфер (доп. задание 6).
void addObject(const glm::vec3& position, const glm::vec3& color) {
	auto& context = graphics::internal::context;

	Object object;
	object.name = "Cube #" + std::to_string(++created_objects);
	object.position = position;
	object.color = color;
	object.uniform_buffer = vkutil::createHostBuffer(sizeof(ObjectUniforms),
	                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	const VkDescriptorSetAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};
	vkAllocateDescriptorSets(context.device, &alloc_info, &object.descriptor_set);

	const VkDescriptorBufferInfo buffer_info = {
		.buffer = object.uniform_buffer.buffer,
		.offset = 0,
		.range = sizeof(ObjectUniforms),
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = object.descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

	objects.push_back(std::move(object));
}

void removeObject(size_t index) {
	auto& context = graphics::internal::context;

	// Буфер может ещё использоваться предыдущим кадром
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeDescriptorSets(context.device, descriptor_pool, 1, &objects[index].descriptor_set);
	vkutil::destroyBuffer(objects[index].uniform_buffer);
	objects.erase(objects.begin() + ptrdiff_t(index));
}

glm::vec3 trajectoryOffset(const Object& object) {
	const float t = object.time;
	switch (object.trajectory) {
	case Trajectory::Circle:
		return { object.radius * std::cos(t), 0.0f, object.radius * std::sin(t) };

	case Trajectory::Lissajous:
		// Пространственная фигура Лиссажу
		return {
			object.radius * std::sin(object.frequency_a * t + float(M_PI) * 0.5f),
			object.height * std::sin((object.frequency_a + object.frequency_b) * t),
			object.radius * std::sin(object.frequency_b * t),
		};

	case Trajectory::Rose: {
		// Роза (r = R * cos(k * t)) в горизонтальной плоскости + колебания по вертикали
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

glm::mat4 projectionMatrix() {
	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	glm::mat4 result;
	if (projection == Projection::Perspective) {
		result = glm::perspective(glm::radians(fov_degrees), aspect, near_plane, far_plane);
	} else {
		const float half_height = ortho_height * 0.5f;
		const float half_width = half_height * aspect;
		result = glm::ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane);
	}

	// В Vulkan ось Y в пространстве отсечения направлена вниз
	result[1][1] *= -1.0f;
	return result;
}

void drawObjectEditor(Object& object) {
	ImGui::PushID(&object);

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", &object.position.x, 0.05f);
	ImGui::DragFloat3("Rotation", &object.rotation.x, 1.0f, -360.0f, 360.0f, "%.0f deg");
	ImGui::DragFloat3("Scale", &object.scale.x, 0.02f, 0.05f, 10.0f);
	if (ImGui::Button("Reset transform")) {
		object.position = glm::vec3(0.0f);
		object.rotation = glm::vec3(0.0f);
		object.scale = glm::vec3(1.0f);
	}

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Color", &object.color.x);
	ImGui::Checkbox("Multiply by procedural vertex colors", &object.vertex_colors);

	ImGui::SeparatorText("Animation");
	ImGui::Checkbox("Move along trajectory", &object.animate);
	const char* trajectories[] = { "Circle", "Lissajous figure", "Rose" };
	ImGui::Combo("Trajectory", reinterpret_cast<int*>(&object.trajectory), trajectories, 3);
	ImGui::SliderFloat("Speed", &object.speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &object.radius, 0.0f, 6.0f);
	if (object.trajectory != Trajectory::Circle) {
		ImGui::SliderFloat("Height", &object.height, 0.0f, 3.0f);
		ImGui::SliderFloat(object.trajectory == Trajectory::Rose ? "Petals (k)" : "Frequency A",
		                   &object.frequency_a, 1.0f, 7.0f, "%.0f");
	}
	if (object.trajectory == Trajectory::Lissajous) {
		ImGui::SliderFloat("Frequency B", &object.frequency_b, 1.0f, 7.0f, "%.0f");
	}
	ImGui::DragFloat3("Spin (deg/s)", &object.spin.x, 1.0f, -720.0f, 720.0f);
	if (ImGui::Button("Restart trajectory")) {
		object.time = 0.0f;
	}

	ImGui::PopID();
}

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab #1: 3D basics (cube)");

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::RadioButton("Perspective", reinterpret_cast<int*>(&projection), int(Projection::Perspective));
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", reinterpret_cast<int*>(&projection), int(Projection::Orthographic));

		if (projection == Projection::Perspective) {
			ImGui::SliderFloat("FOV", &fov_degrees, 10.0f, 120.0f, "%.0f deg");
		} else {
			ImGui::SliderFloat("View height", &ortho_height, 1.0f, 30.0f);
		}
		ImGui::DragFloatRange2("Near / far", &near_plane, &far_plane, 0.05f, 0.01f, 500.0f);
		ImGui::DragFloat3("Camera position", &camera_position.x, 0.05f);
		ImGui::DragFloat3("Camera target", &camera_target.x, 0.05f);
		ImGui::ColorEdit3("Background", &clear_color.x);
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}
		ImGui::SameLine();
		ImGui::TextDisabled("(applies to all objects)");
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add cube")) {
			addObject(glm::vec3(0.0f), glm::vec3(1.0f));
			selected_object = int(objects.size()) - 1;
		}
		ImGui::SameLine();
		ImGui::BeginDisabled(objects.empty());
		if (ImGui::Button("Remove selected")) {
			removeObject(size_t(selected_object));
			selected_object = std::max(0, std::min(selected_object, int(objects.size()) - 1));
		}
		ImGui::EndDisabled();

		if (ImGui::BeginListBox("##objects", ImVec2(-FLT_MIN, 4 * ImGui::GetTextLineHeightWithSpacing()))) {
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

	ImGui::End();
}

} // namespace

bool initialize() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &set_layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		return false;
	}

	constexpr uint32_t max_objects = 64;

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = max_objects,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		return false;
	}

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		return false;
	}

	VkShaderModule vertex_shader = vkutil::loadShaderModule(LAB_SHADER_DIR "cube.vert.spv");
	VkShaderModule fragment_shader = vkutil::loadShaderModule(LAB_SHADER_DIR "cube.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		return false;
	}

	vkutil::PipelineConfig config;
	config.vertex_shader = vertex_shader;
	config.fragment_shader = fragment_shader;
	config.bindings = { { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX } };
	config.attributes = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
	};
	config.layout = pipeline_layout;
	config.render_pass = context.render_pass;

	pipeline = vkutil::createGraphicsPipeline(config);

	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (pipeline == VK_NULL_HANDLE) {
		return false;
	}

	createCubeGeometry();

	// Сцена: центральный куб, движущийся по траектории, и два статичных куба
	addObject(glm::vec3(0.0f), glm::vec3(1.0f));
	objects.back().animate = true;
	objects.back().scale = glm::vec3(1.2f);

	addObject(glm::vec3(-3.5f, 0.0f, -2.0f), glm::vec3(1.0f, 0.6f, 0.3f));
	objects.back().rotation = glm::vec3(20.0f, 35.0f, 0.0f);
	objects.back().vertex_colors = false;

	addObject(glm::vec3(3.5f, 0.0f, -2.0f), glm::vec3(0.4f, 0.8f, 1.0f));
	objects.back().scale = glm::vec3(0.8f, 1.8f, 0.8f);

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

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	vkutil::shutdown();
}

void update(double time) {
	const float dt = float(time - previous_time);
	previous_time = time;

	if (animation_playing) {
		for (auto& object : objects) {
			if (object.animate) {
				object.time += dt * object.speed;
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

	// После prepare() GPU гарантированно закончил предыдущий кадр,
	// поэтому uniform буферы можно безопасно перезаписывать.
	const glm::mat4 view = glm::lookAt(camera_position, camera_target, glm::vec3(0.0f, 1.0f, 0.0f));
	const glm::mat4 view_projection = projectionMatrix() * view;

	for (auto& object : objects) {
		const ObjectUniforms uniforms = {
			.mvp = view_projection * modelMatrix(object),
			.color = glm::vec4(object.color, 1.0f),
			.params = glm::vec4(object.vertex_colors ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f),
		};
		std::memcpy(object.uniform_buffer.mapped, &uniforms, sizeof(uniforms));
	}

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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	// Один и тот же меш рисуется несколько раз, меняется только набор дескрипторов
	for (const auto& object : objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &object.descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
