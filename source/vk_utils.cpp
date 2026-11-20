#include "vk_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "graphics_internal.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244 4100)
#endif // _MSC_VER
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

namespace vkutil {

namespace {

VkCommandPool upload_command_pool = VK_NULL_HANDLE;

} // namespace

void shutdown() {
	auto& context = graphics::internal::context;
	if (upload_command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(context.device, upload_command_pool, nullptr);
		upload_command_pool = VK_NULL_HANDLE;
	}
}

VkShaderModule loadShaderModule(const std::string& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	std::vector<uint32_t> code((size + 3) / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return module;
}

void immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
	auto& context = graphics::internal::context;

	if (upload_command_pool == VK_NULL_HANDLE) {
		const VkCommandPoolCreateInfo pool_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			.queueFamilyIndex = context.graphics_queue_index,
		};
		vkCreateCommandPool(context.device, &pool_info, nullptr, &upload_command_pool);
	}

	const VkCommandBufferAllocateInfo alloc_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = upload_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(context.device, &alloc_info, &cmd);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);
	record(cmd);
	vkEndCommandBuffer(cmd);

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};

	vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeCommandBuffers(context.device, upload_command_pool, 1, &cmd);
}

Buffer createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	};

	Buffer result;
	VmaAllocationInfo info;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result.buffer, &result.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create host visible buffer\n";
		return {};
	}

	result.mapped = info.pMappedData;
	result.size = size;
	return result;
}

Buffer createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
	auto& context = graphics::internal::context;

	Buffer staging = createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	std::memcpy(staging.mapped, data, size_t(size));

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	};

	Buffer result;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result.buffer, &result.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device local buffer\n";
		destroyBuffer(staging);
		return {};
	}
	result.size = size;

	immediateSubmit([&](VkCommandBuffer cmd) {
		const VkBufferCopy region = { .size = size };
		vkCmdCopyBuffer(cmd, staging.buffer, result.buffer, 1, &region);
	});

	destroyBuffer(staging);
	return result;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
                  VkImageAspectFlags aspect, uint32_t mip_levels) {
	auto& context = graphics::internal::context;

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { width, height, 1 },
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	Image result;
	if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
	                   &result.image, &result.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create image\n";
		return {};
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = result.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {
			.aspectMask = aspect,
			.baseMipLevel = 0,
			.levelCount = mip_levels,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &view_info, nullptr, &result.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view\n";
	}

	result.format = format;
	result.width = width;
	result.height = height;
	result.mip_levels = mip_levels;
	return result;
}

Image createTexture(const void* rgba_pixels, uint32_t width, uint32_t height, VkFormat format) {
	auto& context = graphics::internal::context;

	// Mip-уровни генерируются блитами, поэтому формат должен поддерживать линейную фильтрацию при блите
	VkFormatProperties properties;
	vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &properties);
	const bool can_blit = (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) &&
	                      (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
	                      (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);

	const uint32_t mip_levels = can_blit
		? uint32_t(std::floor(std::log2(double(std::max(width, height))))) + 1
		: 1;

	Image image = createImage(width, height, format,
	                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
	                          VK_IMAGE_ASPECT_COLOR_BIT, mip_levels);

	const VkDeviceSize size = VkDeviceSize(width) * height * 4;
	Buffer staging = createHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	std::memcpy(staging.mapped, rgba_pixels, size_t(size));

	immediateSubmit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image.image,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1 },
		};

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &barrier);

		const VkBufferImageCopy region = {
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
			.imageExtent = { width, height, 1 },
		};

		vkCmdCopyBufferToImage(cmd, staging.buffer, image.image,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// Каждый mip-уровень получается уменьшением предыдущего в 2 раза
		int32_t mip_width = int32_t(width);
		int32_t mip_height = int32_t(height);

		for (uint32_t level = 1; level < mip_levels; ++level) {
			barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1 };
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			                     0, 0, nullptr, 0, nullptr, 1, &barrier);

			const int32_t next_width = std::max(mip_width / 2, 1);
			const int32_t next_height = std::max(mip_height / 2, 1);

			const VkImageBlit blit = {
				.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
				.srcOffsets = { { 0, 0, 0 }, { mip_width, mip_height, 1 } },
				.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
				.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } },
			};

			vkCmdBlitImage(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			               image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			               1, &blit, VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			                     0, 0, nullptr, 0, nullptr, 1, &barrier);

			mip_width = next_width;
			mip_height = next_height;
		}

		barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mip_levels - 1, 1, 0, 1 };
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &barrier);
	});

	destroyBuffer(staging);
	return image;
}

bool loadTexture(const std::string& path, bool srgb, Image& out_image) {
	int width, height, channels;
	stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr) {
		std::cerr << "Failed to load image " << path << ": " << stbi_failure_reason() << '\n';
		return false;
	}

	out_image = createTexture(pixels, uint32_t(width), uint32_t(height),
	                          srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);

	stbi_image_free(pixels);
	return out_image.image != VK_NULL_HANDLE;
}

void destroyImage(Image& image) {
	auto& context = graphics::internal::context;
	if (image.view != VK_NULL_HANDLE) {
		vkDestroyImageView(context.device, image.view, nullptr);
	}
	if (image.image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, image.image, image.allocation);
	}
	image = {};
}

VkPipeline createGraphicsPipeline(const PipelineConfig& config) {
	auto& context = graphics::internal::context;

	VkPipelineShaderStageCreateInfo stages[2] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = config.vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = config.fragment_shader,
			.pName = "main",
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = uint32_t(config.bindings.size()),
		.pVertexBindingDescriptions = config.bindings.data(),
		.vertexAttributeDescriptionCount = uint32_t(config.attributes.size()),
		.pVertexAttributeDescriptions = config.attributes.data(),
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = config.topology,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = config.cull_mode,
		.frontFace = config.front_face,
		.depthBiasEnable = config.dynamic_depth_bias ? VK_TRUE : VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = config.depth_test ? VK_TRUE : VK_FALSE,
		.depthWriteEnable = config.depth_write ? VK_TRUE : VK_FALSE,
		.depthCompareOp = config.depth_compare,
	};

	const bool use_render_pass = config.render_pass != VK_NULL_HANDLE;
	const uint32_t color_attachment_count = use_render_pass ? 1 : uint32_t(config.color_formats.size());

	std::vector<VkPipelineColorBlendAttachmentState> blend_attachments(color_attachment_count, {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	});

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = color_attachment_count,
		.pAttachments = blend_attachments.data(),
	};

	std::vector<VkDynamicState> dynamic_states = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	if (config.dynamic_depth_bias) {
		dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
	}

	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(dynamic_states.size()),
		.pDynamicStates = dynamic_states.data(),
	};

	// Для динамического рендеринга вместо render pass указываются форматы вложений
	const VkPipelineRenderingCreateInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = uint32_t(config.color_formats.size()),
		.pColorAttachmentFormats = config.color_formats.data(),
		.depthAttachmentFormat = config.depth_format,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = use_render_pass ? nullptr : &rendering,
		.stageCount = config.fragment_shader != VK_NULL_HANDLE ? 2u : 1u,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic,
		.layout = config.layout,
		.renderPass = config.render_pass,
		.subpass = 0,
	};

	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
	                              nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return VK_NULL_HANDLE;
	}

	return pipeline;
}

} // namespace vkutil
