cmake_minimum_required(VERSION 3.20)
include(FetchContent)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(vulkan-labs LANGUAGES C CXX)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG 3.5.1
)

FetchContent_Declare(
	vk-bootstrap
	GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
	GIT_TAG v1.4.321
)

FetchContent_Declare(
	VulkanMemoryAllocator
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
	GIT_TAG v3.4.0
)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.92.9
)

# Математика (матрицы, векторы)
FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm.git
	GIT_TAG 1.0.1
)

# Загрузка изображений (ЛР №3, №4)
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG 9600522de53973336fa033a289858a4354cbb04c
)

find_package(Vulkan REQUIRED)

set(GLFW_LIBRARY_TYPE STATIC)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLM_BUILD_TESTS OFF)
set(GLM_BUILD_INSTALL OFF)

FetchContent_MakeAvailable(
	glfw
	vk-bootstrap
	VulkanMemoryAllocator
	imgui
	glm
	stb
)

find_program(GLSLC_FOUND glslc REQUIRED)

# Общий каркас: окно, swapchain, ImGui, вспомогательные функции Vulkan.
set(COMMON_SOURCES
	${CMAKE_SOURCE_DIR}/source/main.cpp
	${CMAKE_SOURCE_DIR}/source/graphics_internal.cpp
	${CMAKE_SOURCE_DIR}/source/vk_utils.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/imgui_demo.cpp
)

# add_lab(<имя> <заголовок окна> [DYNAMIC_RENDERING])
#
# Каждая лабораторная работа — отдельный исполняемый файл, который собирается
# из общего каркаса и файлов в каталоге labs/<имя>. Шейдеры из labs/<имя>/shaders
# компилируются в .spv рядом с исходниками.
function(add_lab LAB_NAME LAB_TITLE)
	cmake_parse_arguments(LAB "DYNAMIC_RENDERING" "" "" ${ARGN})

	set(LAB_DIR ${CMAKE_SOURCE_DIR}/labs/${LAB_NAME})
	file(GLOB LAB_SOURCES CONFIGURE_DEPENDS ${LAB_DIR}/*.cpp ${LAB_DIR}/*.hpp)
	file(GLOB LAB_SHADERS CONFIGURE_DEPENDS ${LAB_DIR}/shaders/*.vert ${LAB_DIR}/shaders/*.frag)

	add_executable(${LAB_NAME} ${COMMON_SOURCES} ${LAB_SOURCES})

	if(MSVC)
		target_compile_options(${LAB_NAME} PRIVATE /W4 /wd4201 /utf-8 /EHsc)
		target_link_options(${LAB_NAME} PRIVATE /subsystem:windows /entry:mainCRTStartup)
	else()
		target_compile_options(${LAB_NAME} PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
		if(WIN32)
			target_link_options(${LAB_NAME} PRIVATE -mwindows)
		endif()
	endif()

	set_target_properties(${LAB_NAME} PROPERTIES
		CXX_STANDARD_REQUIRED TRUE
		CXX_STANDARD 20
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	)

	target_compile_definitions(${LAB_NAME} PRIVATE
		_USE_MATH_DEFINES
		GLM_FORCE_DEPTH_ZERO_TO_ONE
		GLM_FORCE_RADIANS
		GLM_ENABLE_EXPERIMENTAL
		LAB_WINDOW_TITLE="${LAB_TITLE}"
		LAB_SHADER_DIR="${LAB_DIR}/shaders/"
		ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets/"
	)

	if(LAB_DYNAMIC_RENDERING)
		target_compile_definitions(${LAB_NAME} PRIVATE GRAPHICS_ENABLE_DYNAMIC_RENDERING)
	endif()

	target_include_directories(${LAB_NAME} PRIVATE
		${CMAKE_SOURCE_DIR}/source
		${LAB_DIR}
		${imgui_SOURCE_DIR}
		${imgui_SOURCE_DIR}/backends
		${stb_SOURCE_DIR}
	)

	target_link_libraries(${LAB_NAME} PRIVATE
		glfw
		Vulkan::Vulkan
		vk-bootstrap::vk-bootstrap
		GPUOpen::VulkanMemoryAllocator
		glm::glm
	)

	set(SHADER_BINARIES)
	foreach(SHADER_SOURCE ${LAB_SHADERS})
		set(SHADER_BINARY ${SHADER_SOURCE}.spv)
		get_filename_component(SHADER_FILE ${SHADER_SOURCE} NAME)
		add_custom_command(
			OUTPUT ${SHADER_BINARY}
			COMMAND ${GLSLC_FOUND} ${SHADER_SOURCE} -o ${SHADER_BINARY}
			DEPENDS ${SHADER_SOURCE}
			COMMENT "Compiling ${LAB_NAME}/${SHADER_FILE} shader"
		)
		list(APPEND SHADER_BINARIES ${SHADER_BINARY})
	endforeach()

	add_custom_target(${LAB_NAME}-shaders DEPENDS ${SHADER_BINARIES} SOURCES ${LAB_SHADERS})
	add_dependencies(${LAB_NAME} ${LAB_NAME}-shaders)
endfunction()

add_lab(lab1 "Lab 1 - 3D basics (variant 1: cube)")
add_lab(lab2 "Lab 2 - Camera and lighting")
add_lab(lab3 "Lab 3 - Texture mapping")
add_lab(lab4 "Lab 4 - Shadow mapping" DYNAMIC_RENDERING)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT lab4)
