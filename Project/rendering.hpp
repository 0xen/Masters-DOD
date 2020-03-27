#pragma once

#define VK_USE_PLATFORM_WIN32_KHR

#include <SDL.h>
#include <SDL_syswm.h>

// Helper Libary I created for vulkan
#include <VkCore.hpp>
#include <VkInitializers.hpp>

class Renderer
{
	static std::unique_ptr<Renderer> sm_singlton;
public:
	static Renderer& Singlton();

private:
	Renderer() {}
	
public:
	


	void WindowSetup(const char* title, int width, int height);
	void PollWindow();
	void DestroyWindow();

	void CreateSurface();

	void InitVulkan(const char* title, uint32_t width, uint32_t height);
	void Render();
	void DestroyVulkan();


	void CreateRenderResources();
	void DestroyRenderResources();
	void RebuildRenderResources();
	void BuildCommandBuffers(std::unique_ptr<VkCommandBuffer>& command_buffers, const uint32_t buffer_count);
};