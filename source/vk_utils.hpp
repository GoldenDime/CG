#pragma once

// Небольшой набор вспомогательных функций поверх Vulkan/VMA, общий для всех
// лабораторных работ. Какие функции используются в какой работе:
//   ЛР №1-2: loadShaderModule, createHostBuffer, createGraphicsPipeline
//   ЛР №3:   + createDeviceBuffer (VRAM через staging), loadTexture
//   ЛР №4:   + createImage (карта теней), конвейер для динамического рендеринга

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

namespace vkutil {

void shutdown();

VkShaderModule loadShaderModule(const std::string& path);

// Выполняет команды и дожидается их завершения (для загрузки ресурсов).
void immediateSubmit(const std::function<void(VkCommandBuffer)>& record);

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

// Буфер в памяти, видимой CPU (постоянно отображён в адресное пространство).
Buffer createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage);

// Буфер в VRAM. Данные копируются командой GPU из временного staging буфера.
Buffer createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);

void destroyBuffer(Buffer& buffer);

struct Image {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VkImageView view = VK_NULL_HANDLE;
	VkFormat format = VK_FORMAT_UNDEFINED;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t mip_levels = 1;
};

Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                  VkImageAspectFlags aspect, uint32_t mip_levels = 1);

// Создаёт текстуру из RGBA8 пикселей: загрузка в VRAM через staging и генерация mip-уровней.
Image createTexture(const void* rgba_pixels, uint32_t width, uint32_t height, VkFormat format);

// Загружает изображение с диска (PNG/JPEG/BMP/TGA/...) при помощи stb_image.
bool loadTexture(const std::string& path, bool srgb, Image& out_image);

void destroyImage(Image& image);

struct PipelineConfig {
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE; // может отсутствовать (проход теней)

	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkCullModeFlags cull_mode = VK_CULL_MODE_BACK_BIT;
	VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	bool depth_test = true;
	bool depth_write = true;
	VkCompareOp depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
	bool dynamic_depth_bias = false;

	VkPipelineLayout layout = VK_NULL_HANDLE;

	// Либо обычный render pass...
	VkRenderPass render_pass = VK_NULL_HANDLE;
	// ...либо форматы вложений для динамического рендеринга (render_pass == VK_NULL_HANDLE)
	std::vector<VkFormat> color_formats;
	VkFormat depth_format = VK_FORMAT_UNDEFINED;
};

VkPipeline createGraphicsPipeline(const PipelineConfig& config);

} // namespace vkutil
