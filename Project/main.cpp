
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <memory>
#include <stdexcept>


#include <thread>
#include <condition_variable>
#include <future>
#include <algorithm>
#include <array>



#define THREE_D 0
#define VISUALISER 1
#define SPHERE_TO_SPHERE_COLLISION 1
#define SPHERE_DEATH 1;
#define SPHERE_COUNT 12800
#define TARGET_FPS 30
#define WORKER_COUNT 128
#define AREA_SLICE_COUNT 128
#define DELTA_TIME 1


#if VISUALISER
// When a frame is about to be rendered, sort the arrays based on the lookup tables to resolve the alpha sorting issues
// Preformance hit, but inproves visualization
#define RESOLVE_ALPHA_SORTING 1
#endif

// Now far can the camera move in and out
const float kCameraRange[2] = { -100.0f,-1000.0f };
// Color min and max range
const float kColorRange[2] = { 0.05f,1.0f };
// Min max range of the sphere spawn
const float kSpawnArea[2] = { -1000.0f,1000.0f };
// Min max distance a sphere could move in the walled area
const float kWallArea[2] = { -1500.0f,1500.0f };

const float kAreaWidth = kWallArea[1] - kWallArea[0];
// Sphere min max scale
const float kSphereScale[2] = { 1.0f,5.0f };
// Sphere min max start velocity
const float kSphereVelocity[2] = { -5.0f,5.0f };
// Sphere start hp
const int kSphereHP = 50;
// How much each laser hit deals to the spheres
const int kLaserDamage = 20;

const float kAreaSliceWidth = kAreaWidth / AREA_SLICE_COUNT;


float AreaSliceBoundariesStart[AREA_SLICE_COUNT];
float AreaSliceBoundariesEnd[AREA_SLICE_COUNT];

// Each worker is allocated a array for each area slice.
// Each worker will loop through there spheres and see if they fall withn the boundary
int WorkerAreaBoundaryChecksStart[WORKER_COUNT][AREA_SLICE_COUNT];
int WorkerAreaBoundaryChecksEnd[WORKER_COUNT][AREA_SLICE_COUNT];

int SphereBoundariesStart[AREA_SLICE_COUNT];
int SphereBoundariesEnd[AREA_SLICE_COUNT];

// Move the compile time sphere count into a valid platform unsigned int
const uint32_t sphere_count = SPHERE_COUNT;

const uint32_t NumWorkers = WORKER_COUNT;

const unsigned sphere_worker_groups = sphere_count / NumWorkers;

// These letters will make up the names of the spheres
const char name_letters[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

std::thread threads[NumWorkers];
std::condition_variable work_ready[NumWorkers];
std::mutex worker_mutex[NumWorkers];
bool have_work[NumWorkers];
bool can_render;
std::condition_variable render_work_ready;
float deltaTime;


// Used to store 
unsigned int current_lockup_table = 0;
unsigned int SphereLookupTable[2][sphere_count];

struct WorkerTask
{
	unsigned int offset;
	unsigned int count;
	std::packaged_task<void(unsigned int, unsigned int, unsigned int)> task;
};

WorkerTask tasks[NumWorkers];

#if THREE_D
// All X positions
// All Y positions
// All Z positions
// All Sphere Rads
const unsigned int sphere_position_array_length = sphere_count * 4;
const unsigned int sphere_velocity_array_size = sphere_count * 3;
#else
// All X positions
// All Y positions
// All Sphere Rads
const unsigned int sphere_position_array_length = sphere_count * 3;
const unsigned int sphere_velocity_array_size = sphere_count * 2;
#endif


constexpr unsigned int GetNumberOfDigits(unsigned int i)
{
	int digits = 0; 
	while (i != 0) { i /= 16; digits++; }
	return digits;
}

constexpr unsigned int sphere_name_length = GetNumberOfDigits(sphere_count);
const unsigned int sphere_name_array_size = sphere_name_length * sphere_count;
const unsigned int sphere_color_array_size = sphere_count * 4;


static union
{
	struct
	{
		float X[sphere_count];
		float Y[sphere_count];
#if THREE_D
		float Z[sphere_count];
#endif
		float Scale[sphere_count];
	}spheres;
	float SphereData[sphere_position_array_length]{ 0.0f };
};
char sphere_names[sphere_name_array_size]{ 0 };

static union
{
	struct
	{
		float X[sphere_count];
		float Y[sphere_count];
#if THREE_D
		float Z[sphere_count];
#endif
	}velocity;
	float sphere_velocity[sphere_velocity_array_size]{ 0 };
};

int sphere_hp[sphere_count]{ 0 };

float sphere_color[sphere_color_array_size]{ 0 };


float line_position[6]{ 0 };

// If the worker fonud a sphere that the line collides with, how far from the start was it
float line_sphere_colision_distance[NumWorkers];
// If we found a sphere that the line collided with, what is its non lookup table id
int line_sphere_index[NumWorkers];

#include <SDL.h>
#include <SDL_syswm.h>
#if VISUALISER
// Helper Libary I created for vulkan
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <VkCore.hpp>
#include <VkInitializers.hpp>

#else
// This type def comes with vulkan, if we dont have it normaly, include it here
typedef unsigned int uint32_t;
#endif



__forceinline void WaitForWorkers()
{
	for (int i = 0; i < NumWorkers; ++i)
	{
		std::unique_lock<std::mutex> lock(worker_mutex[i]);
		while (have_work[i])
		{// Wait until work is done
			work_ready[i].wait(lock);
		}
	}
}

void Worker(unsigned int id)
{
	srand(id);
	while (true)
	{
		{
			// Guard use of haveWork from other thread
			std::unique_lock<std::mutex> lock(worker_mutex[id]);
			while (!have_work[id])
			{ // Wait for some work
				work_ready[id].wait(lock);
			};
		}

		tasks[id].task(id, tasks[id].offset, tasks[id].count);


		{
			std::unique_lock<std::mutex> lock(worker_mutex[id]);
			have_work[id] = false;
		}


		work_ready[id].notify_one(); // Tell main thread
	}
}

__forceinline void StartTask(std::function<void(unsigned int, unsigned int, unsigned int)> funcPtr, unsigned int perWorker, unsigned int workerCount = NumWorkers)
{
	for (int i = 0; i < workerCount; ++i)
	{
		tasks[i].offset = i * perWorker;
		tasks[i].count = perWorker;
		tasks[i].task = std::packaged_task<void(unsigned int, unsigned int, unsigned int)>(funcPtr);
		{ // Only use haveWork if other thread is not
			std::unique_lock<std::mutex> lock(worker_mutex[i]);
			have_work[i] = true;
		}
		work_ready[i].notify_one(); // Tell worker
	}
}
Uint64 delta_time;
//std::chrono::steady_clock::time_point last;
float GetDeltaTime()
{
	Uint64 now = SDL_GetPerformanceCounter();
	float temp = static_cast<float>((float)(now - delta_time) / SDL_GetPerformanceFrequency());
	delta_time = now;
	return temp;

	/*std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();


	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(now - last);

	last = now;

	return time_span.count();*/

}

#if VISUALISER

const float kTargetFPS = 1.0f / TARGET_FPS;

std::thread renderer;
std::thread sync;
std::mutex renderer_lock;

__forceinline void WaitForRender()
{
	std::unique_lock<std::mutex> lock(renderer_lock);
	while (can_render)
	{// Wait until work is done
		render_work_ready.wait(lock);
	}
}

struct VertexData
{
	float position[3];
	float uv[2];
	float padding[3];
};


const unsigned int verticies_count = 4;
const unsigned int indicies_count = 6;

VertexData verticies[verticies_count] =
{
	{{-1.0f,-1.0f,0.0f},{-1.0f,-1.0f}},
	{{1.0f,1.0f,0.0f},{1.0f,1.0f}},
	{{-1.0f,1.0f,0.0f},{-1.0f,1.0f}},
	{ {1.0f,-1.0f,0.0f},{1.0f,-1.0f}}
};

uint32_t indicies[indicies_count] = {
	0,1,2,
	0,3,1,
};


VkDeviceSize vertex_buffer_size = sizeof(VertexData) * verticies_count;
VkDeviceSize index_buffer_size = sizeof(uint32_t) * indicies_count;


void CreateRenderResources();
void DestroyRenderResources();
void RebuildRenderResources();
void BuildCommandBuffers(std::unique_ptr<VkCommandBuffer>& command_buffers, const uint32_t buffer_count);

VkInstance instance;
VkDebugReportCallbackEXT debugger;
VkPhysicalDevice physical_device = VK_NULL_HANDLE;


uint32_t physical_devices_queue_family = 0;
VkPhysicalDeviceProperties physical_device_properties;
VkPhysicalDeviceFeatures physical_device_features;
VkPhysicalDeviceMemoryProperties physical_device_mem_properties;

VkDevice device = VK_NULL_HANDLE;
VkQueue graphics_queue = VK_NULL_HANDLE;
VkQueue present_queue = VK_NULL_HANDLE;
VkCommandPool command_pool;


float camera_z = kCameraRange[0];


bool window_open;
SDL_Window* window;
SDL_SysWMinfo window_info;
VkSurfaceCapabilitiesKHR surface_capabilities;
VkSurfaceKHR surface;
uint32_t window_width;
uint32_t window_height;

VkSurfaceFormatKHR surface_format;
VkPresentModeKHR present_mode;

VkSwapchainKHR swap_chain;
std::unique_ptr<VkImage> swapchain_images;
uint32_t swapchain_image_count;
std::unique_ptr<VkImageView> swapchain_image_views;

VkRenderPass renderpass = VK_NULL_HANDLE;
std::unique_ptr<VkFramebuffer> framebuffers = nullptr;
std::unique_ptr<VkHelper::VulkanAttachments> framebuffer_attachments = nullptr;

uint32_t current_frame_index = 0; // What frame are we currently using
std::unique_ptr<VkFence> fences = nullptr;

VkSemaphore image_available_semaphore;
VkSemaphore render_finished_semaphore;
VkSubmitInfo sumbit_info = {};
VkPresentInfoKHR present_info = {};
VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

std::unique_ptr<VkCommandBuffer> graphics_command_buffers = nullptr;


VkPipeline sphere_graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout graphics_pipeline_layout = VK_NULL_HANDLE;
const uint32_t sphere_shader_count = 2;
std::unique_ptr<VkShaderModule> sphere_shader_modules = nullptr;

VkPipeline line_graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout line_pipeline_layout = VK_NULL_HANDLE;
const uint32_t line_shader_count = 2;
std::unique_ptr<VkShaderModule> line_shader_modules = nullptr;


VkBuffer vertex_buffer = VK_NULL_HANDLE;
VkDeviceMemory vertex_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* vertex_mapped_buffer_memory = nullptr;


VkBuffer index_buffer = VK_NULL_HANDLE;
VkDeviceMemory index_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* index_mapped_buffer_memory = nullptr;


// Next we need to define how many bytes of data we want to reserve on the GPU
const VkDeviceSize position_buffer_size = sizeof(float) * sphere_position_array_length;

// Next we need to define how many bytes of data we want to reserve on the GPU
const VkDeviceSize color_buffer_size = sizeof(float) * sphere_color_array_size;

// Create a storage variable for the buffer and buffer memory
VkBuffer position_buffer = VK_NULL_HANDLE;
VkDeviceMemory position_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* position_mapped_buffer_memory = nullptr;

// Next we need to define how many bytes of data we want to reserve on the GPU
// We are going to be passing the sphere count and the aspect ratio
const VkDeviceSize settings_buffer_size = sizeof(uint32_t) * 3;

// Create a storage variable for the buffer and buffer memory
VkBuffer settings_buffer = VK_NULL_HANDLE;
VkDeviceMemory settings_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* settings_mapped_buffer_memory = nullptr;

// Next we need to define how many bytes of data we want to reserve on the GPU
// We are going to be passing the sphere count and the aspect ratio
const VkDeviceSize line_buffer_size = sizeof(uint32_t) * 6;

// Create a storage variable for the buffer and buffer memory
VkBuffer color_buffer = VK_NULL_HANDLE;
VkDeviceMemory color_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* color_mapped_buffer_memory = nullptr;

// Create a storage variable for the buffer and buffer memory
VkBuffer line_buffer = VK_NULL_HANDLE;
VkDeviceMemory line_buffer_memory = VK_NULL_HANDLE;
// Raw pointer that will point to GPU memory
void* line_mapped_buffer_memory = nullptr;

VkDescriptorPool position_descriptor_pool;
VkDescriptorSetLayout position_descriptor_set_layout;
VkDescriptorSet position_descriptor_set;

VkDescriptorPool line_descriptor_pool;
VkDescriptorSetLayout line_descriptor_set_layout;
VkDescriptorSet line_descriptor_set;

void WindowSetup(const char* title, int width, int height)
{
	window = SDL_CreateWindow(
		title,
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height,
		SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
	);
	SDL_ShowWindow(window);
	window_open = true;

	window_width = width;
	window_height = height;

	SDL_VERSION(&window_info.version);
	bool sucsess = SDL_GetWindowWMInfo(window, &window_info);
	assert(sucsess && "Error, unable to get window info");
}

void UpdateGPUSettings(uint32_t sphere_count, float aspectRatio)
{
	uint32_t data[3]{ sphere_count };

	memcpy(
		&data[1],
		&aspectRatio,
		sizeof(uint32_t)
	);

	memcpy(
		&data[2],
		&camera_z,
		sizeof(uint32_t)
	);

	memcpy(
		settings_mapped_buffer_memory,                                         // The destination for our memory (GPU)
		data,                                                                  // Source for the memory (CPU-Ram)
		settings_buffer_size                                                   // How much data we are transfering
	);
}

void PollWindow()
{

	// Poll Window
	SDL_Event event;
	bool rebuild = false;
	while (SDL_PollEvent(&event) > 0)
	{
		switch (event.type)
		{
		case SDL_QUIT:
			window_open = false;
			break;
		case SDL_MOUSEWHEEL:
			camera_z += event.wheel.y*50;
			if (camera_z > kCameraRange[0])camera_z = kCameraRange[0];
			if (camera_z < kCameraRange[1])camera_z = kCameraRange[1];
			UpdateGPUSettings(sphere_count, (float)window_width / (float)window_height);
			break;
		case SDL_WINDOWEVENT:
			switch (event.window.event)
			{
				//Get new dimensions and repaint on window size change
			case SDL_WINDOWEVENT_SIZE_CHANGED:

				window_width = event.window.data1;
				window_height = event.window.data2;

				break;
			}
			break;
			//
		}
	}
}
void DestroyWindow()
{
	SDL_DestroyWindow(window);
}

void CreateSurface()
{
	auto CreateWin32SurfaceKHR = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");

	VkWin32SurfaceCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
	createInfo.hwnd = window_info.info.win.window;
	createInfo.hinstance = GetModuleHandle(nullptr);

	if (!CreateWin32SurfaceKHR || CreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}


// Everything within the Setup is from previous tuturials
// Setup
// - Instance
// - Debugger
// - Physical Device
// - Device
// - Command Pool
// - Buffer
void InitVulkan()
{

	// Define what Layers and Extentions we require
	const uint32_t extention_count = 3;
	const char *instance_extensions[extention_count] = { "VK_EXT_debug_report" ,VK_KHR_SURFACE_EXTENSION_NAME,VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
	const uint32_t layer_count = 1;
	const char *instance_layers[layer_count] = { "VK_LAYER_LUNARG_standard_validation" };

	// Check to see if we have the layer requirments
	assert(VkHelper::CheckLayersSupport(instance_layers, 1) && "Unsupported Layers Found");

	// Create the Vulkan Instance
	instance = VkHelper::CreateInstance(
		instance_extensions, extention_count,
		instance_layers, layer_count,
		"2 - Device", VK_MAKE_VERSION(1, 0, 0),
		"Vulkan", VK_MAKE_VERSION(1, 0, 0),
		VK_MAKE_VERSION(1, 1, 108));

	// Attach a debugger to the application to give us validation feedback.
	// This is usefull as it tells us about any issues without application
	debugger = VkHelper::CreateDebugger(instance);

	// Define what Device Extentions we require
	const uint32_t physical_device_extention_count = 1;
	// Note that this extention list is diffrent from the instance on as we are telling the system what device settings we need.
	const char *physical_device_extensions[physical_device_extention_count] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };



	// The surface creation is added here as it needs to happen before the physical device creation and after we have said to the instance that we need a surface
	// The surface is the refrence back to the OS window
	CreateSurface();



	// Find a physical device for us to use
	bool foundPhysicalDevice = VkHelper::GetPhysicalDevice(
		instance,
		physical_device,                                       // Return of physical device instance
		physical_device_properties,                            // Physical device properties
		physical_devices_queue_family,                         // Physical device queue family
		physical_device_features,                              // Physical device features
		physical_device_mem_properties,                        // Physical device memory properties
		physical_device_extensions,                            // What extentions out device needs to have
		physical_device_extention_count,                       // Extention count
		VK_QUEUE_GRAPHICS_BIT,                                 // What queues we need to be avaliable
		surface                                                // Pass the instance to the OS monitor
	);

	// Make sure we found a physical device
	assert(foundPhysicalDevice);

	// Define how many queues we will need in our project, for now, we will just create a single queue
	static const float queue_priority = 1.0f;
	VkDeviceQueueCreateInfo queue_create_info = VkHelper::DeviceQueueCreateInfo(
		&queue_priority,
		1,
		physical_devices_queue_family
	);


	// Now we have the physical device, create the device instance
	device = VkHelper::CreateDevice(
		physical_device,                                       // The physical device we are basic the device from
		&queue_create_info,                                    // A pointer array, pointing to a list of queues we want to make
		1,                                                     // How many queues are in the list
		physical_device_features,                              // What features do you want enabled on the device
		physical_device_extensions,                            // What extentions do you want on the device
		physical_device_extention_count                        // How many extentions are there
	);

	vkGetDeviceQueue(
		device,
		physical_devices_queue_family,
		0,
		&graphics_queue
	);

	vkGetDeviceQueue(
		device,
		physical_devices_queue_family,
		0,
		&present_queue
	);

	command_pool = VkHelper::CreateCommandPool(
		device,
		physical_devices_queue_family,                         // What queue family we are wanting to use to send commands to the GPU
		VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT        // Allows any commands we create, the ability to be reset. This is helpfull as we wont need to
	);                                                         // keep allocating new commands, we can reuse them


	CreateRenderResources();

	fences = std::unique_ptr<VkFence>(new VkFence[swapchain_image_count]);

	for (unsigned int i = 0; i < swapchain_image_count; i++)
	{
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		VkResult create_fence_result = vkCreateFence(
			device,
			&info,
			nullptr,
			&fences.get()[i]
		);
		assert(create_fence_result == VK_SUCCESS);
	}


	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkResult create_semaphore_result = vkCreateSemaphore(device,
		&semaphore_create_info,
		nullptr,
		&image_available_semaphore
	);
	assert(create_semaphore_result == VK_SUCCESS);

	create_semaphore_result = vkCreateSemaphore(device,
		&semaphore_create_info,
		nullptr,
		&render_finished_semaphore
	);
	assert(create_semaphore_result == VK_SUCCESS);


	VkCommandBufferAllocateInfo command_buffer_allocate_info = VkHelper::CommandBufferAllocateInfo(
		command_pool,
		swapchain_image_count
	);

	graphics_command_buffers = std::unique_ptr<VkCommandBuffer>(new VkCommandBuffer[swapchain_image_count]);

	VkResult allocate_command_buffer_resut = vkAllocateCommandBuffers(
		device,
		&command_buffer_allocate_info,
		graphics_command_buffers.get()
	);
	assert(allocate_command_buffer_resut == VK_SUCCESS);

	sumbit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	sumbit_info.waitSemaphoreCount = 1;
	sumbit_info.pWaitSemaphores = &image_available_semaphore;
	sumbit_info.pWaitDstStageMask = &wait_stages;
	sumbit_info.commandBufferCount = 1;
	sumbit_info.signalSemaphoreCount = 1;
	sumbit_info.pSignalSemaphores = &render_finished_semaphore;

	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	// The swapchain will be recreated whenevr the window is resized or the KHR becomes invalid
	// But the pointer to our swapchain will remain intact
	present_info.pSwapchains = &swap_chain;
	present_info.pResults = nullptr;


	bool buffer_created = VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		position_buffer,                                                 // What buffer are we going to be creating
		position_buffer_memory,                                          // The output for the buffer memory
		position_buffer_size,                                            // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                              // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	VkResult mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		position_buffer_memory,                                         // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		position_buffer_size,                                           // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&position_mapped_buffer_memory                                  // The return for the memory pointer
	);


	// Could we map the GPU memory to our CPU accessable pointer
	assert(mapped_memory_result == VK_SUCCESS);

	buffer_created = VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		color_buffer,                                                 // What buffer are we going to be creating
		color_buffer_memory,                                          // The output for the buffer memory
		color_buffer_size,                                            // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                              // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		color_buffer_memory,                                         // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		color_buffer_size,                                           // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&color_mapped_buffer_memory                                  // The return for the memory pointer
	);


	// Could we map the GPU memory to our CPU accessable pointer
	assert(mapped_memory_result == VK_SUCCESS);

	buffer_created = VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		line_buffer,                                                     // What buffer are we going to be creating
		line_buffer_memory,                                              // The output for the buffer memory
		line_buffer_size,                                                // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,                              // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		line_buffer_memory,                                             // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		line_buffer_size,                                               // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&line_mapped_buffer_memory                                      // The return for the memory pointer
	);


	// Could we map the GPU memory to our CPU accessable pointer
	assert(mapped_memory_result == VK_SUCCESS);

	buffer_created = VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		settings_buffer,                                                 // What buffer are we going to be creating
		settings_buffer_memory,                                          // The output for the buffer memory
		settings_buffer_size,                                            // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,                              // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		settings_buffer_memory,                                         // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		settings_buffer_size,                                           // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&settings_mapped_buffer_memory                                  // The return for the memory pointer
	);


	// Could we map the GPU memory to our CPU accessable pointer
	assert(mapped_memory_result == VK_SUCCESS);


	UpdateGPUSettings(sphere_count, (float)window_width / (float)window_height);


	{ // Create the sphere descriptor set
		const uint32_t descriptor_pool_size_count = 3;

		VkDescriptorPoolSize pool_size[descriptor_pool_size_count] = {
			VkHelper::DescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
			VkHelper::DescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
			VkHelper::DescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
		};

		position_descriptor_pool = VkHelper::CreateDescriptorPool(device, pool_size, 1, 10);

		VkDescriptorSetLayoutBinding layout_bindings[descriptor_pool_size_count] = {
			VkHelper::DescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT),
			VkHelper::DescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT),
			VkHelper::DescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		position_descriptor_set_layout = VkHelper::CreateDescriptorSetLayout(device, layout_bindings, descriptor_pool_size_count);

		position_descriptor_set = VkHelper::AllocateDescriptorSet(device, position_descriptor_pool, position_descriptor_set_layout, 1);

		{ // Update the Descriptor Set

			VkDescriptorBufferInfo buffer_infos[3];

			buffer_infos[0] = {};
			buffer_infos[0].buffer = settings_buffer;
			buffer_infos[0].range = settings_buffer_size;
			buffer_infos[0].offset = 0;

			buffer_infos[1] = {};
			buffer_infos[1].buffer = position_buffer;
			buffer_infos[1].range = position_buffer_size;
			buffer_infos[1].offset = 0;

			buffer_infos[2] = {};
			buffer_infos[2].buffer = color_buffer;
			buffer_infos[2].range = color_buffer_size;
			buffer_infos[2].offset = 0;

			VkWriteDescriptorSet position_descriptor_write_set[3];
			position_descriptor_write_set[0] = {};
			position_descriptor_write_set[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			position_descriptor_write_set[0].dstSet = position_descriptor_set;
			position_descriptor_write_set[0].dstBinding = 0;
			position_descriptor_write_set[0].dstArrayElement = 0;
			position_descriptor_write_set[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			position_descriptor_write_set[0].descriptorCount = 1;
			position_descriptor_write_set[0].pBufferInfo = &buffer_infos[0];

			position_descriptor_write_set[1] = {};
			position_descriptor_write_set[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			position_descriptor_write_set[1].dstSet = position_descriptor_set;
			position_descriptor_write_set[1].dstBinding = 1;
			position_descriptor_write_set[1].dstArrayElement = 0;
			position_descriptor_write_set[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			position_descriptor_write_set[1].descriptorCount = 1;
			position_descriptor_write_set[1].pBufferInfo = &buffer_infos[1];

			position_descriptor_write_set[2] = {};
			position_descriptor_write_set[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			position_descriptor_write_set[2].dstSet = position_descriptor_set;
			position_descriptor_write_set[2].dstBinding = 2;
			position_descriptor_write_set[2].dstArrayElement = 0;
			position_descriptor_write_set[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			position_descriptor_write_set[2].descriptorCount = 1;
			position_descriptor_write_set[2].pBufferInfo = &buffer_infos[2];

			vkUpdateDescriptorSets(
				device,
				3,
				position_descriptor_write_set,
				0,
				NULL
			);
		}

	}
	{ // Create the line descriptor set
		const uint32_t descriptor_pool_size_count = 1;

		VkDescriptorPoolSize pool_size[descriptor_pool_size_count] = {
			VkHelper::DescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
		};

		line_descriptor_pool = VkHelper::CreateDescriptorPool(device, pool_size, 1, 10);

		VkDescriptorSetLayoutBinding layout_bindings[descriptor_pool_size_count] = {
			VkHelper::DescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT)
		};

		line_descriptor_set_layout = VkHelper::CreateDescriptorSetLayout(device, layout_bindings, descriptor_pool_size_count);

		line_descriptor_set = VkHelper::AllocateDescriptorSet(device, line_descriptor_pool, line_descriptor_set_layout, 1);

		{ // Update the Descriptor Set

			VkDescriptorBufferInfo buffer_infos[descriptor_pool_size_count];

			buffer_infos[0] = {};
			buffer_infos[0].buffer = settings_buffer;
			buffer_infos[0].range = settings_buffer_size;
			buffer_infos[0].offset = 0;

			VkWriteDescriptorSet line_descriptor_write_set[descriptor_pool_size_count];
			line_descriptor_write_set[0] = {};
			line_descriptor_write_set[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			line_descriptor_write_set[0].dstSet = line_descriptor_set;
			line_descriptor_write_set[0].dstBinding = 0;
			line_descriptor_write_set[0].dstArrayElement = 0;
			line_descriptor_write_set[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			line_descriptor_write_set[0].descriptorCount = 1;
			line_descriptor_write_set[0].pBufferInfo = &buffer_infos[0];


			vkUpdateDescriptorSets(
				device,
				descriptor_pool_size_count,
				line_descriptor_write_set,
				0,
				NULL
			);
		}

	}

	


}

// Everything within the Destroy is from previous tuturials
// Destroy
// - Buffer
// - Command Pool
// - Device
// - Debugger
// - Instance
void DestroyVulkan()
{
	DestroyRenderResources();

	for (uint32_t i = 0; i < swapchain_image_count; i++)
	{
		vkDestroyFence(
			device,
			fences.get()[i],
			nullptr
		);
	}

	vkDestroyDescriptorSetLayout(
		device,
		position_descriptor_set_layout,
		nullptr
	);
	vkDestroyDescriptorPool(
		device,
		position_descriptor_pool,
		nullptr
	);

	{ // Destroy the possition buffer
		// Now we unmap the data
		vkUnmapMemory(
			device,
			position_buffer_memory
		);

		// Clean up the buffer data
		vkDestroyBuffer(
			device,
			position_buffer,
			nullptr
		);

		// Free the memory that was allocated for the buffer
		vkFreeMemory(
			device,
			position_buffer_memory,
			nullptr
		);
	}

	{ // Destroy the color buffer
		// Now we unmap the data
		vkUnmapMemory(
			device,
			color_buffer_memory
		);

		// Clean up the buffer data
		vkDestroyBuffer(
			device,
			color_buffer,
			nullptr
		);

		// Free the memory that was allocated for the buffer
		vkFreeMemory(
			device,
			color_buffer_memory,
			nullptr
		);
	}

	{ // Destroy the settings buffer
		// Now we unmap the data
		vkUnmapMemory(
			device,
			settings_buffer_memory
		);

		// Clean up the buffer data
		vkDestroyBuffer(
			device,
			settings_buffer,
			nullptr
		);

		// Free the memory that was allocated for the buffer
		vkFreeMemory(
			device,
			settings_buffer_memory,
			nullptr
		);
	}

	vkDestroySemaphore(
		device,
		image_available_semaphore,
		nullptr
	);

	vkDestroySemaphore(
		device,
		render_finished_semaphore,
		nullptr
	);


	// Clean up the command pool
	vkDestroyCommandPool(
		device,
		command_pool,
		nullptr
	);

	// Clean up the device now that the project is stopping
	vkDestroyDevice(
		device,
		nullptr
	);

	// Destroy the debug callback
	// We cant directly call vkDestroyDebugReportCallbackEXT as we need to find the pointer within the Vulkan DLL, See function inplmentation for details.
	VkHelper::DestroyDebugger(
		instance,
		debugger
	);

	// Clean up the vulkan instance
	vkDestroyInstance(
		instance,
		NULL
	);

	DestroyWindow();
}


void CreateRenderResources()
{
	swap_chain = VkHelper::CreateSwapchain(
		physical_device,
		device,
		surface,
		surface_capabilities,
		surface_format,
		present_mode,
		window_width,
		window_height,
		swapchain_image_count,
		swapchain_images,
		swapchain_image_views
	);


	const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
	renderpass = VkHelper::CreateRenderPass(
		physical_device,
		device,
		surface_format.format,
		colorFormat,
		swapchain_image_count,
		physical_device_mem_properties,
		physical_device_features,
		physical_device_properties,
		command_pool,
		graphics_queue,
		window_width,
		window_height,
		framebuffers,
		framebuffer_attachments,
		swapchain_image_views
	);
}

void DestroyRenderResources()
{

	vkDestroyRenderPass(
		device,
		renderpass,
		nullptr
	);

	vkDestroySwapchainKHR(
		device,
		swap_chain,
		nullptr
	);

	for (uint32_t i = 0; i < swapchain_image_count; i++)
	{
		vkDestroyImageView(
			device,
			swapchain_image_views.get()[i],
			nullptr
		);


		vkDestroyFramebuffer(
			device,
			framebuffers.get()[i],
			nullptr
		);

		vkDestroyImageView(
			device,
			framebuffer_attachments.get()[i].color.view,
			nullptr
		);
		vkDestroyImage(
			device,
			framebuffer_attachments.get()[i].color.image,
			nullptr
		);
		vkFreeMemory(
			device,
			framebuffer_attachments.get()[i].color.memory,
			nullptr
		);
		vkDestroySampler(
			device,
			framebuffer_attachments.get()[i].color.sampler,
			nullptr
		);
		vkDestroyImageView(
			device,
			framebuffer_attachments.get()[i].depth.view,
			nullptr
		);
		vkDestroyImage(
			device,
			framebuffer_attachments.get()[i].depth.image,
			nullptr
		);
		vkFreeMemory(
			device,
			framebuffer_attachments.get()[i].depth.memory,
			nullptr
		);
		vkDestroySampler(
			device,
			framebuffer_attachments.get()[i].depth.sampler,
			nullptr
		);
	}
}


void RebuildRenderResources()
{
	VkResult device_idle_result = vkDeviceWaitIdle(device);
	assert(device_idle_result == VK_SUCCESS);

	DestroyRenderResources();
	CreateRenderResources();
	BuildCommandBuffers(graphics_command_buffers, swapchain_image_count);

	UpdateGPUSettings(sphere_count, (float)window_width / (float)window_height);
}


void BuildCommandBuffers(std::unique_ptr<VkCommandBuffer>& command_buffers, const uint32_t buffer_count)
{
	VkCommandBufferBeginInfo command_buffer_begin_info = VkHelper::CommandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	const float clear_color[4] = { 0.2f,0.2f,0.2f,1.0f };

	VkClearValue clear_values[3]{};

	std::copy(std::begin(clear_color), std::end(clear_color), std::begin(clear_values[0].color.float32)); // Present
	std::copy(std::begin(clear_color), std::end(clear_color), std::begin(clear_values[1].color.float32)); // Color Image
	clear_values[2].depthStencil = { 1.0f, 0 };                                                           // Depth Image


	VkRenderPassBeginInfo render_pass_info = VkHelper::RenderPassBeginInfo(
		renderpass,
		{ 0,0 },
		{ window_width, window_height }
	);

	render_pass_info.clearValueCount = 3;
	render_pass_info.pClearValues = clear_values;

	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	for (unsigned int i = 0; i < buffer_count; i++)
	{
		// Reset the command buffers
		VkResult reset_command_buffer_result = vkResetCommandBuffer(
			command_buffers.get()[i],
			0
		);
		assert(reset_command_buffer_result == VK_SUCCESS);

		render_pass_info.framebuffer = framebuffers.get()[i];

		VkResult begin_command_buffer_result = vkBeginCommandBuffer(
			command_buffers.get()[i],
			&command_buffer_begin_info
		);
		assert(begin_command_buffer_result == VK_SUCCESS);

		vkCmdBeginRenderPass(
			command_buffers.get()[i],
			&render_pass_info,
			VK_SUBPASS_CONTENTS_INLINE
		);


		vkCmdSetLineWidth(
			command_buffers.get()[i],
			1.0f
		);

		VkViewport viewport = VkHelper::Viewport(
			(float)window_width,
			(float)window_height,
			0,
			0,
			0.0f,
			1.0f
		);

		vkCmdSetViewport(
			command_buffers.get()[i],
			0,
			1,
			&viewport
		);

		VkRect2D scissor{};
		scissor.extent.width = window_width;
		scissor.extent.height = window_height;
		scissor.offset.x = 0;
		scissor.offset.y = 0;

		vkCmdSetScissor(
			command_buffers.get()[i],
			0,
			1,
			&scissor
		);

		vkCmdSetLineWidth(
			command_buffers.get()[i],
			1.0f
		);

		vkCmdBindPipeline(
			command_buffers.get()[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			sphere_graphics_pipeline
		);

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(
			command_buffers.get()[i],
			0,
			1,
			&vertex_buffer,
			offsets
		);

		vkCmdBindIndexBuffer(
			command_buffers.get()[i],
			index_buffer,
			0,
			VK_INDEX_TYPE_UINT32
		);

		vkCmdBindDescriptorSets(
			command_buffers.get()[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			graphics_pipeline_layout,
			0,
			1,
			&position_descriptor_set,
			0,
			NULL
		);

		vkCmdDrawIndexed(
			command_buffers.get()[i],
			indicies_count,
			sphere_count,
			0,
			0,
			0
		);




		vkCmdBindPipeline(
			command_buffers.get()[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			line_graphics_pipeline
		);



		vkCmdBindVertexBuffers(
			command_buffers.get()[i],
			0,
			1,
			&line_buffer,
			offsets
		);


		vkCmdBindDescriptorSets(
			command_buffers.get()[i],
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			line_pipeline_layout,
			0,
			1,
			&line_descriptor_set,
			0,
			NULL
		);

		vkCmdDraw(
			command_buffers.get()[i],
			2,
			1,
			0,
			0
		);
















		/*vkCmdDrawIndexedIndirect(
			command_buffers.get()[i],
			m_indirect_draw_buffer->GetBufferData(BufferSlot::Primary)->buffer,
			j * sizeof(VkDrawIndexedIndirectCommand),
			1,
			sizeof(VkDrawIndexedIndirectCommand));*/


		vkCmdEndRenderPass(
			command_buffers.get()[i]
		);

		VkResult end_command_buffer_result = vkEndCommandBuffer(
			command_buffers.get()[i]
		);
		assert(end_command_buffer_result == VK_SUCCESS);

	}
}

void Render()
{
	// Find next image
	VkResult wait_for_fences = vkWaitForFences(
		device,
		1,
		&fences.get()[current_frame_index],
		VK_TRUE,
		UINT32_MAX
	);
	assert(wait_for_fences == VK_SUCCESS);

	VkResult acquire_next_image_result = vkAcquireNextImageKHR(
		device,
		swap_chain,
		UINT64_MAX,
		image_available_semaphore,
		VK_NULL_HANDLE,
		&current_frame_index
	);
	assert(acquire_next_image_result == VK_SUCCESS);

	// Reset and wait
	VkResult queue_idle_result = vkQueueWaitIdle(
		graphics_queue
	);
	assert(queue_idle_result == VK_SUCCESS);

	VkResult reset_fences_result = vkResetFences(
		device,
		1,
		&fences.get()[current_frame_index]
	);
	assert(reset_fences_result == VK_SUCCESS);

	sumbit_info.pCommandBuffers = &graphics_command_buffers.get()[current_frame_index];

	VkResult queue_submit_result = vkQueueSubmit(
		graphics_queue,
		1,
		&sumbit_info,
		fences.get()[current_frame_index]
	);
	assert(queue_submit_result == VK_SUCCESS);

	present_info.pImageIndices = &current_frame_index;

	VkResult queue_present_result = vkQueuePresentKHR(
		present_queue,
		&present_info
	);
	// If the window was resized or something else made the current render invalid, we need to rebuild all the
	// render resources
	if (queue_present_result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		RebuildRenderResources();
	}

	assert(queue_present_result == VK_SUCCESS || queue_present_result == VK_ERROR_OUT_OF_DATE_KHR);


	VkResult device_idle_result = vkDeviceWaitIdle(device);
	assert(device_idle_result == VK_SUCCESS);
}

void CreatePipelines()
{
	{ // Sphere Graphics Pipeline
		const uint32_t vertex_input_attribute_description_count = 2;
		std::unique_ptr<VkVertexInputAttributeDescription> vertex_input_attribute_descriptions =
			std::unique_ptr<VkVertexInputAttributeDescription>(new VkVertexInputAttributeDescription[vertex_input_attribute_description_count]);

		// Used to store the position data of the vertex
		vertex_input_attribute_descriptions.get()[0].binding = 0;
		vertex_input_attribute_descriptions.get()[0].location = 0;
		vertex_input_attribute_descriptions.get()[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertex_input_attribute_descriptions.get()[0].offset = offsetof(VertexData, position);

		vertex_input_attribute_descriptions.get()[1].binding = 0;
		vertex_input_attribute_descriptions.get()[1].location = 1;
		vertex_input_attribute_descriptions.get()[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertex_input_attribute_descriptions.get()[1].offset = offsetof(VertexData, uv);

		const uint32_t vertex_input_binding_description_count = 1;

		std::unique_ptr<VkVertexInputBindingDescription> vertex_input_binding_descriptions =
			std::unique_ptr<VkVertexInputBindingDescription>(new VkVertexInputBindingDescription[vertex_input_binding_description_count]);

		vertex_input_binding_descriptions.get()[0].binding = 0;
		vertex_input_binding_descriptions.get()[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertex_input_binding_descriptions.get()[0].stride = sizeof(VertexData);


		const char* shader_paths[sphere_shader_count]{

	#if THREE_D
			"../../Project/Shaders/Sphere/3D/vert.spv",
	#else
			"../../Project/Shaders/Sphere/2D/vert.spv",
	#endif
			"../../Project/Shaders/Sphere/frag.spv"
		};

		VkShaderStageFlagBits shader_stages_bits[sphere_shader_count]{
			VK_SHADER_STAGE_VERTEX_BIT,
			VK_SHADER_STAGE_FRAGMENT_BIT
		};
		sphere_shader_modules = std::unique_ptr<VkShaderModule>(new VkShaderModule[sphere_shader_count]);

		const uint32_t dynamic_state_count = 3;

		VkDynamicState dynamic_states[dynamic_state_count] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
			VK_DYNAMIC_STATE_LINE_WIDTH
		};

		const uint32_t descriptor_set_layout_count = 1;

		VkDescriptorSetLayout descriptor_set_layout[descriptor_set_layout_count] = {
			position_descriptor_set_layout
		};

		sphere_graphics_pipeline = VkHelper::CreateGraphicsPipeline(
			physical_device,
			device,
			renderpass,
			graphics_pipeline_layout,
			sphere_shader_count,
			shader_paths,
			shader_stages_bits,
			sphere_shader_modules.get(),
			descriptor_set_layout_count,
			descriptor_set_layout,
			vertex_input_attribute_description_count,
			vertex_input_attribute_descriptions.get(),
			vertex_input_binding_description_count,
			vertex_input_binding_descriptions.get(),
			dynamic_state_count,
			dynamic_states
		);
	}
	{ // Line Graphics Pipeline
		const uint32_t vertex_input_attribute_description_count = 1;
		std::unique_ptr<VkVertexInputAttributeDescription> vertex_input_attribute_descriptions =
			std::unique_ptr<VkVertexInputAttributeDescription>(new VkVertexInputAttributeDescription[vertex_input_attribute_description_count]);

		// Used to store the position data of the vertex
		vertex_input_attribute_descriptions.get()[0].binding = 0;
		vertex_input_attribute_descriptions.get()[0].location = 0;
		vertex_input_attribute_descriptions.get()[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertex_input_attribute_descriptions.get()[0].offset = 0;

		const uint32_t vertex_input_binding_description_count = 1;

		std::unique_ptr<VkVertexInputBindingDescription> vertex_input_binding_descriptions =
			std::unique_ptr<VkVertexInputBindingDescription>(new VkVertexInputBindingDescription[vertex_input_binding_description_count]);

		vertex_input_binding_descriptions.get()[0].binding = 0;
		vertex_input_binding_descriptions.get()[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		vertex_input_binding_descriptions.get()[0].stride = sizeof(float) * 3;

		const char* shader_paths[line_shader_count]{
			"../../Project/Shaders/Line/vert.spv",
			"../../Project/Shaders/Line/frag.spv"
		};

		VkShaderStageFlagBits shader_stages_bits[sphere_shader_count]{
			VK_SHADER_STAGE_VERTEX_BIT,
			VK_SHADER_STAGE_FRAGMENT_BIT
		};
		line_shader_modules = std::unique_ptr<VkShaderModule>(new VkShaderModule[line_shader_count]);

		const uint32_t dynamic_state_count = 3;

		VkDynamicState dynamic_states[dynamic_state_count] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
			VK_DYNAMIC_STATE_LINE_WIDTH
		};

		const uint32_t descriptor_set_layout_count = 1;

		VkDescriptorSetLayout descriptor_set_layout[descriptor_set_layout_count] = {
			line_descriptor_set_layout
		};

		line_graphics_pipeline = VkHelper::CreateGraphicsPipeline(
			physical_device,
			device,
			renderpass,
			line_pipeline_layout,
			sphere_shader_count,
			shader_paths,
			shader_stages_bits,
			line_shader_modules.get(),
			descriptor_set_layout_count,
			descriptor_set_layout,
			vertex_input_attribute_description_count,
			vertex_input_attribute_descriptions.get(),
			vertex_input_binding_description_count,
			vertex_input_binding_descriptions.get(),
			dynamic_state_count,
			dynamic_states,
			VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
			VK_POLYGON_MODE_FILL,
			1.0f,
			VK_CULL_MODE_BACK_BIT,
			VK_FALSE,
			VK_FALSE
		);
	}
	
}


void DestroyPipelines()
{
	vkDestroyPipeline(
		device,
		sphere_graphics_pipeline,
		nullptr
	);

	vkDestroyPipelineLayout(
		device,
		graphics_pipeline_layout,
		nullptr
	);

	for (uint32_t i = 0; i < sphere_shader_count; i++)
	{
		vkDestroyShaderModule(
			device,
			sphere_shader_modules.get()[i],
			nullptr
		);
	}
}

void CreateModelBuffers()
{

	const unsigned int buffer_size = sizeof(float) * 1000;
	// Vertex buffer
	VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		vertex_buffer,                                                   // What buffer are we going to be creating
		vertex_buffer_memory,                                            // The output for the buffer memory
		vertex_buffer_size,                                              // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,                            // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	VkResult vertex_mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		vertex_buffer_memory,                                           // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		vertex_buffer_size,                                             // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&vertex_mapped_buffer_memory                                    // The return for the memory pointer
	);

	// Could we map the GPU memory to our CPU accessable pointer
	assert(vertex_mapped_memory_result == VK_SUCCESS);

	// First we copy the example data to the GPU
	memcpy(
		vertex_mapped_buffer_memory,                                    // The destination for our memory (GPU)
		verticies,                                                      // Source for the memory (CPU-Ram)
		vertex_buffer_size                                              // How much data we are transfering
	);



	// Vertex buffer
	VkHelper::CreateBuffer(
		device,                                                          // What device are we going to use to create the buffer
		physical_device_mem_properties,                                  // What memory properties are avaliable on the device
		index_buffer,                                                    // What buffer are we going to be creating
		index_buffer_memory,                                             // The output for the buffer memory
		index_buffer_size,                                               // How much memory we wish to allocate on the GPU
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,                            // What type of buffer do we want. Buffers can have multiple types, for example, uniform & vertex buffer.
																		 // for now we want to keep the buffer spetilised to one type as this will allow vulkan to optimize the data.
		VK_SHARING_MODE_EXCLUSIVE,                                       // There are two modes, exclusive and concurrent. Defines if it can concurrently be used by multiple queue
																		 // families at the same time
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT                              // What properties do we rquire of our memory
	);

	// Get the pointer to the GPU memory
	VkResult index_mapped_memory_result = vkMapMemory(
		device,                                                         // The device that the memory is on
		index_buffer_memory,                                            // The device memory instance
		0,                                                              // Offset from the memorys start that we are accessing
		index_buffer_size,                                              // How much memory are we accessing
		0,                                                              // Flags (we dont need this for basic buffers)
		&index_mapped_buffer_memory                                     // The return for the memory pointer
	);

	// Could we map the GPU memory to our CPU accessable pointer
	assert(index_mapped_memory_result == VK_SUCCESS);

	// First we copy the example data to the GPU
	memcpy(
		index_mapped_buffer_memory,                                           // The destination for our memory (GPU)
		indicies,                                                             // Source for the memory (CPU-Ram)
		index_buffer_size                                                     // How much data we are transfering
	);

}

void DestroyModelBuffers()
{
	// Now we unmap the data
	vkUnmapMemory(
		device,
		vertex_buffer_memory
	);

	// Clean up the buffer data
	vkDestroyBuffer(
		device,
		vertex_buffer,
		nullptr
	);

	// Free the memory that was allocated for the buffer
	vkFreeMemory(
		device,
		vertex_buffer_memory,
		nullptr
	);


	// Now we unmap the data
	vkUnmapMemory(
		device,
		index_buffer_memory
	);

	// Clean up the buffer data
	vkDestroyBuffer(
		device,
		index_buffer,
		nullptr
	);

	// Free the memory that was allocated for the buffer
	vkFreeMemory(
		device,
		index_buffer_memory,
		nullptr
	);
}


void RendererThread()
{
	while (window_open)
	{
		{
			// Guard use of haveWork from other thread
			std::unique_lock<std::mutex> lock(renderer_lock);
			while (!can_render)
			{ // Wait for some work
				render_work_ready.wait(lock);
			};
		}

		Render();

		{
			std::unique_lock<std::mutex> lock(renderer_lock);
			can_render = false;
		}


		render_work_ready.notify_one(); // Tell main thread
	}

}



void StartRenderer()
{
	WindowSetup("DOD", 1080, 720);
	InitVulkan();
	CreateModelBuffers();
	CreatePipelines();
	BuildCommandBuffers(graphics_command_buffers, swapchain_image_count);

	{// temp
		line_position[0] = -40.0f;
		line_position[1] = 0.0f;
		line_position[2] = 0.0f;
		line_position[3] = 40.0f;
		line_position[4] = 0.0f;
		line_position[5] = 0.0f;
		memcpy(
			line_mapped_buffer_memory,                                           // The destination for our memory (GPU)
			line_position,                                                              // Source for the memory (CPU-Ram)
			line_buffer_size                                                     // How much data we are transfering
		);
	}
	memcpy(
		position_mapped_buffer_memory,                                           // The destination for our memory (GPU)
		SphereData,                                                              // Source for the memory (CPU-Ram)
		position_buffer_size                                                     // How much data we are transfering
	);
	memcpy(
		color_mapped_buffer_memory,                                              // The destination for our memory (GPU)
		sphere_color,                                                            // Source for the memory (CPU-Ram)
		color_buffer_size                                                        // How much data we are transfering
	);

	renderer = std::thread(RendererThread);
}


void WorkerAlphaSort(unsigned int worker_id, unsigned int offset, unsigned int count)
{

	unsigned int currentLookup;
	float* position_buffer = reinterpret_cast<float*>(position_mapped_buffer_memory);
	float* color_buffer = reinterpret_cast<float*>(color_mapped_buffer_memory);


	// Loop unraveled transfer of colors to the GPU

	unsigned int max = offset + count;
	unsigned int color_max = max * 4;
	for (unsigned int i = offset * 4, j = offset; i < color_max;)
	{
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
		currentLookup = SphereLookupTable[current_lockup_table][j++] * 4;
		color_buffer[i++] = sphere_color[currentLookup++]; // R
		color_buffer[i++] = sphere_color[currentLookup++]; // G
		color_buffer[i++] = sphere_color[currentLookup++]; // B
		++i; // Alpha generated on GPU
	}


	// X position loop unraveled
	for (unsigned int i = offset; i < max;)
	{
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++] = spheres.X[SphereLookupTable[current_lockup_table][i]];
	}
	// Y position loop unraveled
	for (unsigned int i = offset; i < max;)
	{
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + sphere_count] = spheres.Y[SphereLookupTable[current_lockup_table][i]];
	}
	unsigned int position_offset = sphere_count * 2;
#if THREE_D
	// Z position loop unraveled
	for (unsigned int i = offset; i < max;)
	{
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Z[SphereLookupTable[current_lockup_table][i]];
	}
	position_offset += sphere_count;
#endif
	// Scale loop unraveled
	for (unsigned int i = offset; i < max;)
	{
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
		position_buffer[i++ + position_offset] = spheres.Scale[SphereLookupTable[current_lockup_table][i]];
	}

}

#endif



__forceinline float Random(float min, float max)
{
	return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

void GenerateNewRay()
{
	line_position[0] = Random(kSpawnArea[0], kSpawnArea[1]);
	line_position[1] = Random(kSpawnArea[0], kSpawnArea[1]);
	line_position[2] = Random(kSpawnArea[0], kSpawnArea[1]);
	line_position[3] = Random(kSpawnArea[0], kSpawnArea[1]);
#if THREE_D
	line_position[4] = Random(kSpawnArea[0], kSpawnArea[1]);
	line_position[5] = Random(kSpawnArea[0], kSpawnArea[1]);
#else
	line_position[4] = 0.0f;
	line_position[5] = 0.0f;
#endif

#if VISUALISER
	memcpy(
		line_mapped_buffer_memory,                                           // The destination for our memory (GPU)
		line_position,                                                              // Source for the memory (CPU-Ram)
		line_buffer_size                                                     // How much data we are transfering
	);
#endif
}


void FindClosestLineIntersections(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	// Get the diffrence between the two end points of all the 
	float lineDistX = line_position[0] - line_position[3];
	float lineDistY = line_position[1] - line_position[4];

#if THREE_D
	float lineDistZ = line_position[2] - line_position[5];
#endif

	// Calculate the lines full length
	float llen = sqrt(
		(lineDistX*lineDistX)
		+ (lineDistY*lineDistY)
#if THREE_D
		+ (lineDistZ*lineDistZ)
#endif;
	);

	// Gethalf the length of the line
	float llenHalf = llen / 2;

	float lineCenterX = lineDistX / 2;
	float lineCenterY = lineDistY / 2;
#if THREE_D
	float lineCenterZ = lineDistZ / 2;
#endif;

	// precalculate the line length to the power of 2
	float linePow = pow(llen, 2);
	unsigned int max = offset + count;

	line_sphere_index[worker_id] = -1;
	line_sphere_colision_distance[worker_id] = 0.0f;

	float startSphereXDiffrence, startSphereYDiffrence, lengthToStart;

	unsigned int currentSphereLookup;

	for (int i = offset; i < max; i++)
	{
		currentSphereLookup = SphereLookupTable[current_lockup_table][i];
#if SPHERE_DEATH
		if (sphere_hp[currentSphereLookup] <= 0)
			continue;
#endif
		// If we have already found a sphere, check to see if we are closer to its stating position
		// Due to branch prediction, the CPU will auto asume the result of this based on previous results making this check marganaly slow
		if (line_sphere_index[worker_id] >= 0)
		{
			startSphereXDiffrence = line_position[0] - spheres.X[currentSphereLookup];
			startSphereYDiffrence = line_position[1] - spheres.Y[currentSphereLookup];
#if THREE_D
			startSphereZDiffrence = line_position[2] - spheres.Z[currentSphereLookup];
#endif

			lengthToStart = sqrt(
				(startSphereXDiffrence * startSphereXDiffrence)
				+ (startSphereYDiffrence * startSphereYDiffrence)
#if THREE_D
				+ (startSphereZDiffrence * startSphereZDiffrence)
#endif
			);
			// If we are already further away from the start then the last found sphere, break
			if (lengthToStart > line_sphere_colision_distance[worker_id])
				continue;
		}
		// Work out the diffrence between the line center and the sphere
		float startLineToSphereDistX = lineCenterX - spheres.X[currentSphereLookup];
		float startLineToSphereDistY = lineCenterY - spheres.Y[currentSphereLookup];
#if THREE_D
		float startLineToSphereDistZ = lineCenterZ - spheres.Z[currentSphereLookup];
#endif

		float sphereToStartLength = sqrt(
			  (startLineToSphereDistX * startLineToSphereDistX)
			+ (startLineToSphereDistY * startLineToSphereDistY)
#if THREE_D
			+ (startLineToSphereDistZ * startLineToSphereDistZ)
#endif
		);

		// early breakout if the sphere is found to already be too far away
		if (sphereToStartLength > llenHalf)
		{
			continue;
		}





		float dot = (
			  ((spheres.X[currentSphereLookup] - line_position[0]) * (-lineDistX))
			+ ((spheres.Y[currentSphereLookup] - line_position[1]) * (-lineDistY))
#if THREE_D
			+ ((spheres.Z[currentSphereLookup] - line_position[2]) * (-lineDistZ))
#endif
			) / linePow;

		float closestX = line_position[0] + (dot * (-lineDistX));
		float closestY = line_position[1] + (dot * (-lineDistY));
#if THREE_D
		float closestZ = line_position[2] + (dot * (-lineDistZ));
#endif



		float lineToSphereDistX = closestX - spheres.X[currentSphereLookup];
		float lineToSphereDistY = closestY - spheres.Y[currentSphereLookup];
#if THREE_D
		float lineToSphereDistZ = closestZ - spheres.Z[currentSphereLookup];
#endif



		float sphereToClosest = sqrt(
			  (lineToSphereDistX * lineToSphereDistX)
			+ (lineToSphereDistY * lineToSphereDistY)
#if THREE_D
			+ (lineToSphereDistZ * lineToSphereDistZ)
#endif
		);

		if (sphereToClosest < spheres.Scale[currentSphereLookup])
		{
			//std::cout << "Line collided with" << i << std::endl;

			// If we have not found a closer sphere yet
			// Due to branch prediction, the CPU will auto asume the result of this based on previous results making this check marganaly slow
			if (line_sphere_index[worker_id] <0)
			{
				startSphereXDiffrence = line_position[0] - spheres.X[currentSphereLookup];
				startSphereYDiffrence = line_position[1] - spheres.Y[currentSphereLookup];
#if THREE_D
				startSphereZDiffrence = line_position[2] - spheres.Z[currentSphereLookup];
#endif
				lengthToStart = sqrt(
					(startSphereXDiffrence * startSphereXDiffrence)
					+ (startSphereYDiffrence * startSphereYDiffrence)
#if THREE_D
					+ (startSphereZDiffrence * startSphereZDiffrence)
#endif;
				);
			}
			line_sphere_colision_distance[worker_id] = lengthToStart;
			line_sphere_index[worker_id] = i;
		}

	}

}


void WorkerSetupSimulation(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	// For this we setup all sphere data that needs the random function first so that the CPU can cache the function

	{
		unsigned int end = offset + count;
		// Give each sphere a position and scale
		for (unsigned int i = offset; i < end; ++i)
		{
			spheres.X[i] = Random(kSpawnArea[0], kSpawnArea[1]);
			spheres.Y[i] = Random(kSpawnArea[0], kSpawnArea[1]);
#if THREE_D
			spheres.Z[i] = Random(kSpawnArea[0], kSpawnArea[1]);
#endif
			spheres.Scale[i] = Random(kSphereScale[0], kSphereScale[1]);

			sphere_hp[i] = kSphereHP;

			// Setup the lookup table so element 1
			SphereLookupTable[current_lockup_table][i] = i;
		}
	}

	{
		unsigned int start = offset << 2;
		unsigned int end = start + (count << 2);
		// Give each sphere a color
		for (unsigned int i = start; i < end; i += 4)
		{
			sphere_color[i] = Random(kColorRange[0], kColorRange[1]);
			sphere_color[i + 1] = Random(kColorRange[0], kColorRange[1]);
			sphere_color[i + 2] = Random(kColorRange[0], kColorRange[1]);
			sphere_color[i + 3] = 1.0f;
		}
	}

	{
		unsigned int start = offset << 1;
		unsigned int end = start + (count << 1);
		// Give each sphere a color
		for (unsigned int i = start; i < end;)
		{
			sphere_velocity[i++] = Random(kSphereVelocity[0], kSphereVelocity[1]);
			sphere_velocity[i++] = Random(kSphereVelocity[0], kSphereVelocity[1]);
#if THREE_D
			sphere_velocity[i++] = Random(kSphereVelocity[0], kSphereVelocity[1]);
#endif
		}
	}



	// Give each sphere a name
	{
		// Reset the name array
		{
			// Cant use bit shifting here as the sphere name length can change based on the number of spheres
			unsigned int start = offset * sphere_name_length;
			unsigned int end = start + (count * sphere_name_length);
			for (unsigned int i = start; i < end; ++i)
			{
				sphere_names[i] = { '0' };
			}
		}


		// Loop through each sphere and give it a unique name
		{

			unsigned int char_index = 0;
			unsigned int name_letter_index = 0;
			unsigned int names_offset = offset * sphere_name_length;
			unsigned int end = offset + count;
			// Loop through all spheres to give them a name
			for (unsigned int i = offset; i < end; ++i)
			{
				// name_letter_index = What letter of there name are we wrighting, starting from the last letter and moving forward
				// char_index = We use this to get the first 8 bits of the number, convert it to hex and bit shift it to get its next 8

				// Loop through for every 8 bits that make up the number and store its hex value
				for (name_letter_index = sphere_name_length - 1, char_index = i; char_index != 0; char_index >>= 4, --name_letter_index)
				{
					char letter = name_letters[char_index & 0xF];
					sphere_names[names_offset + name_letter_index] = letter;
				}

				names_offset += sphere_name_length;
			}

		}
	}
}

void SetupSimulation()
{
	float sphere_rad = kSphereScale[1] / 2;
	for (unsigned int i = 0; i < AREA_SLICE_COUNT; ++i)
	{
		AreaSliceBoundariesStart[i] = kWallArea[0] + (kAreaSliceWidth * i) - sphere_rad;
		AreaSliceBoundariesEnd[i] = kWallArea[0] + (kAreaSliceWidth * (i + 1)) + sphere_rad;
	}


	// Reset delta time
	delta_time = SDL_GetPerformanceCounter();

	for (unsigned int i = 0; i < NumWorkers; i++)
	{
		threads[i] = std::thread(Worker, i);
	}


	StartTask(WorkerSetupSimulation, sphere_worker_groups);
	WaitForWorkers();
}

void swap(unsigned int* a, unsigned int* b)
{
	unsigned int t = *a;
	*a = *b;
	*b = t;
}

// Quicksort Solution https://www.geeksforgeeks.org/cpp-program-for-quicksort/
__forceinline int QuicksortPartition(int low, int high)
{
#if THREE_D
	float pivot = spheres.Z[SphereLookupTable[current_lockup_table][high]];    // pivot 
#else
	float pivot = spheres.X[SphereLookupTable[current_lockup_table][high]];    // pivot 
#endif
	int i = (low - 1);  // Index of smaller element 

	for (int j = low; j <= high - 1; j++)
	{
		// If current element is smaller than or 
		// equal to pivot 
#if THREE_D
		if (spheres.Z[SphereLookupTable[current_lockup_table][j]] <= pivot)
#else
		if (spheres.X[SphereLookupTable[current_lockup_table][j]] <= pivot)
#endif
		{
			i++;
			swap(&SphereLookupTable[current_lockup_table][i], &SphereLookupTable[current_lockup_table][j]);
		}
	}
	swap(&SphereLookupTable[current_lockup_table][i + 1], &SphereLookupTable[current_lockup_table][high]);
	return (i + 1);
}


void QuickSort(int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		   at right place */
		int pi = QuicksortPartition(low, high);

		// Separately sort elements before 
		// partition and after partition 
		QuickSort(low, pi - 1);
		QuickSort(pi + 1, high);
	}
}

void WorkerQuickSort(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	QuickSort(offset, offset + count - 1);
}

#if THREE_D

inline void Reflect(float & rx, float & ry, float & rz, 
	float vx, float vy, float vz, 
	float nx, float ny, float nz)
{
	float dot = vx * nx + vy * ny + vz * nz;
	rx = vx - 2 * dot * nx;
	ry = vy - 2 * dot * ny;
	rz = vz - 2 * dot * nz;
}
#else

inline void Reflect(float & rx, float & ry, float vx, float vy, float nx, float ny)
{
	float dot = vx * nx + vy * ny;
	rx = vx - 2 * dot * nx;
	ry = vy - 2 * dot * ny;
}
#endif






// Collision detection solution found here https://www.gamasutra.com/view/feature/131424/pool_hall_lessons_fast_accurate_.php?print=1
// Calculate the sphere to sphere calculate for the spheres
void WorkerCollisionDetection(unsigned int worker_id, unsigned int a, unsigned int b)
{

	unsigned int offset = SphereBoundariesStart[worker_id];
	unsigned int count = (SphereBoundariesEnd[worker_id] - SphereBoundariesStart[worker_id]) + 1;
	unsigned int max = offset + count;

	unsigned int s1, s2, sphere1lookup, sphere2lookup;
	// Sphere 1
	float s1vx, s1vy, s1vl, s1x, s1y, s1r;
	// Sphere 2
	float s2x, s2y, s2r;

	// Sphere-To-Sphere components
	float distanceAV, Cx, Cy, lengthC, Nx, Ny, D, F, sumRadiiSquared, T, rx, ry;

#if THREE_D
	float s1vz, s1z, s2z, Cz, Nz, rz;
#endif

	for (s1 = offset; s1 < max; ++s1)
	{
		sphere1lookup = SphereLookupTable[current_lockup_table][s1];
		// Get all of the spheres components that we are checking movment for
		s1vx = velocity.X[sphere1lookup];
		s1vy = velocity.Y[sphere1lookup];
#if THREE_D
		s1vz = velocity.Z[sphere1lookup];
#endif
#if DELTA_TIME
		s1vx *= deltaTime;
		s1vy *= deltaTime;
#if THREE_D
		s1vz *= deltaTime;
#endif
#endif


		// Get the spheres velocity vector length
		s1vl = sqrt(
			s1vx * s1vx + 
#if THREE_D
			s1vz * s1vz +
#endif
			s1vy * s1vy
		);

		s1x = spheres.X[sphere1lookup];
		s1y = spheres.Y[sphere1lookup];
#if THREE_D
		s1z = spheres.Z[sphere1lookup];
#endif
		s1r = spheres.Scale[sphere1lookup];

		for (s2 = offset; s2 < max; ++s2)
		{
			if (s1 == s2)
				continue;
			sphere2lookup = SphereLookupTable[current_lockup_table][s2];
			// Get the spheres scale
			s2r = spheres.Scale[sphere2lookup];
			// Get the total radius amount
			float sumRadii = (s2r + s1r);
			s2y = spheres.Y[sphere2lookup];
			// Since we are sorting the scene along the x or (z if in 3D) axis, if the y axis is already out of the scale range, continue
			if (abs(s2y - s1y) > sumRadii)
				continue;

			// Now load the X axis since we know its feasible that the spheres could collide with the Y axis
			s2x = spheres.X[sphere2lookup];
#if THREE_D
			// Since we are sorting the scene along the z axis, if the x axis is already out of the scale range, continue
			if (abs(s2x - s1x) > sumRadii)
				continue;
			s2z = spheres.Z[sphere2lookup];
#endif







			// Get the distance between the two spheres
			Cx = s2x - s1x;
			Cy = s2y - s1y;
#if THREE_D
			Cz = s2z - s1z;
#endif
			// Get the centere distance length
			lengthC = sqrt(
				Cx * Cx +
#if THREE_D
				Cz * Cz +
#endif
				Cy * Cy
			);
			// Is the length of the velocity vector larger then the distance between the two spheres - there rad
			if (s1vl < lengthC - sumRadii)
				continue;

			// Get the normalised velocity vector
			Nx = s1vx / s1vl;
			Ny = s1vy / s1vl;
#if THREE_D
			Nz = s1vz / s1vl;
#endif

			// Calculate the first part of the length of the closest point along the velocity we get to before touching the sphere
			// We do not do sqrt here as it is expensive and we can do a check before hand to get rid of some invalid cases
			D = Nx * Cx +
#if THREE_D
				Nz * Cz +
#endif
				Ny * Cy;

			if (D <= 0)
				continue;

			F = (lengthC * lengthC) - (D * D);

			sumRadiiSquared = sumRadii * sumRadii;
			if (F >= sumRadiiSquared)
				continue;

			T = sumRadiiSquared - F;

			if (T < 0)
				continue;
			distanceAV = D - sqrt(T);

			// Is the point we found in the range along the velocity vector
			if (s1vl < distanceAV)
				continue;

			// Now we know there must have been a colision!!

			{ // Move the sphere to the surface of the other sphere
				s1vx = Nx * distanceAV;
				spheres.X[sphere1lookup] += s1vx;

				s1vy = Ny * distanceAV;
				spheres.Y[sphere1lookup] += s1vy;

#if THREE_D
				s1vz = Nz * distanceAV;
				spheres.Z[sphere1lookup] += s1vz;
#endif
			}



			{ // Calculate the reflection of the sphere of the other sphere 


			// Recauclulate the vector between the two spheres and normalise it for the reflection calculation
				Cx /= lengthC;
				Cy /= lengthC;
#if THREE_D
				Cz /= lengthC;
				// Cauculate a normalised reflection of the velocity of the other sphere
				Reflect(
					rx, ry, rz,
					velocity.X[sphere1lookup], velocity.Y[sphere1lookup], velocity.Z[sphere1lookup],
					Cx, Cy, Cz
				);
#else
				// Cauculate a normalised reflection of the velocity of the other sphere
				Reflect(
					rx, ry,
					velocity.X[sphere1lookup], velocity.Y[sphere1lookup],
					Cx, Cy
				);
#endif
				
				// Store the velocity back in its array positions for future updates
				velocity.X[sphere1lookup] = rx;
				velocity.Y[sphere1lookup] = ry;
#if THREE_D
				velocity.Z[sphere1lookup] = rz;
#endif
			}

			{ // Preform the final part of the spheres movment once it has bounced of the other sphere
				s1vx = velocity.X[sphere1lookup] * abs(distanceAV - 1.0f);
				s1vy = velocity.Y[sphere1lookup] * abs(distanceAV - 1.0f);
#if THREE_D
				s1vz = velocity.Z[sphere1lookup] * abs(distanceAV - 1.0f);
#endif

#if DELTA_TIME
				s1vx *= deltaTime;
				s1vy *= deltaTime;
#if THREE_D
				s1vz *= deltaTime;
#endif
#endif
			}

			break;
		}
		spheres.X[sphere1lookup] += s1vx;
		if (spheres.X[sphere1lookup] < kWallArea[0] || spheres.X[sphere1lookup] > kWallArea[1])
		{
			velocity.X[sphere1lookup] = -velocity.X[sphere1lookup];
			spheres.X[sphere1lookup] += velocity.X[sphere1lookup];
		}

		spheres.Y[sphere1lookup] += s1vy;
		if (spheres.Y[sphere1lookup] < kWallArea[0] || spheres.Y[sphere1lookup] > kWallArea[1])
		{
			velocity.Y[sphere1lookup] = -velocity.Y[sphere1lookup];
			spheres.Y[sphere1lookup] += velocity.Y[sphere1lookup];
		}
#if THREE_D
		spheres.Z[sphere1lookup] += s1vz;
		if (spheres.Z[sphere1lookup] < kWallArea[0] || spheres.Z[sphere1lookup] > kWallArea[1])
		{
			velocity.Z[sphere1lookup] = -velocity.Z[sphere1lookup];
			spheres.Z[sphere1lookup] += velocity.Z[sphere1lookup];
		}
#endif
	}

}


void WorkerBubbleSort(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	unsigned int max = offset + count;
	unsigned int j = 0;
	unsigned int temp;
	for (unsigned int i = offset; i < max; i++)
	{
		for (j = i + 1; j < max; j++)
		{
#if THREE_D
			if (spheres.Z[SphereLookupTable[current_lockup_table][j]] <
				spheres.Z[SphereLookupTable[current_lockup_table][i]])
#else
			if (spheres.X[SphereLookupTable[current_lockup_table][j]] <
				spheres.X[SphereLookupTable[current_lockup_table][i]])
#endif
			{
				temp = SphereLookupTable[current_lockup_table][i];
				SphereLookupTable[current_lockup_table][i] = SphereLookupTable[current_lockup_table][j];
				SphereLookupTable[current_lockup_table][j] = temp;
			}
		}
	}
}

void WorkerBoundaryFinder(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	// Used for loop index
	unsigned int i = 0;
	// Reset all the found spheres that meet that boundary
	for (i = 0; i < AREA_SLICE_COUNT; ++i)
	{
		WorkerAreaBoundaryChecksStart[worker_id][i] = -1;
		WorkerAreaBoundaryChecksEnd[worker_id][i] = -1;
	}




	// What boundary are we looking for the largest sphere
	unsigned int checkingBoundary = 0;


	unsigned int max = offset + count;
	for (checkingBoundary = 0; checkingBoundary < AREA_SLICE_COUNT; ++checkingBoundary)
	{
		for (i = offset; i < max; ++i)
		{
#if THREE_D
			bool start = spheres.Z[SphereLookupTable[current_lockup_table][i]] > AreaSliceBoundariesStart[checkingBoundary];
			bool end = spheres.Z[SphereLookupTable[current_lockup_table][i]] < AreaSliceBoundariesEnd[checkingBoundary];
#else
			bool start = spheres.X[SphereLookupTable[current_lockup_table][i]] > AreaSliceBoundariesStart[checkingBoundary];
			bool end = spheres.X[SphereLookupTable[current_lockup_table][i]] < AreaSliceBoundariesEnd[checkingBoundary];
#endif
			if (!start && end)
				continue;

			if (start && end)
			{
				if (WorkerAreaBoundaryChecksStart[worker_id][checkingBoundary] < 0)
				{
					WorkerAreaBoundaryChecksStart[worker_id][checkingBoundary] = i;
				}

				WorkerAreaBoundaryChecksEnd[worker_id][checkingBoundary] = i;
			}
			else
			{
				break;
			}
		}
	}

}

void WorkerMergeSortedArray(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	unsigned int half_way = count / 2;

	unsigned int* sorted_array_1 = &SphereLookupTable[current_lockup_table][offset];
	unsigned int* sorted_array_2 = &SphereLookupTable[current_lockup_table][offset + half_way];

	unsigned int* target_array = &SphereLookupTable[(current_lockup_table + 1) % 2][offset];

	int i = 0, j = 0, k = 0;
	while (i < half_way && j < half_way)
	{
		// If we are in 3D we want to sort by the z buffer so we can also resolve the alpha depth order at the same time!
#if THREE_D
		if (spheres.Z[sorted_array_1[i]] < spheres.Z[sorted_array_2[j]])
#else
		if (spheres.X[sorted_array_1[i]] < spheres.X[sorted_array_2[j]])
#endif
			target_array[k++] = sorted_array_1[i++];
		else
			target_array[k++] = sorted_array_2[j++];
	}
	while (i < half_way)
		target_array[k++] = sorted_array_1[i++];
	while (j < half_way)
		target_array[k++] = sorted_array_2[j++];

}

void WorkerSimulateSphere(unsigned int worker_id, unsigned int offset, unsigned int count)
{
	unsigned int end = offset + count;

	float* AxisPtr = spheres.X + offset;
	// X axis
	for (uint32_t i = offset; i < end; )
	{
#if DELTA_TIME
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++] * deltaTime;
#else
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
		*(AxisPtr++) += sphere_velocity[i++];
#endif
	}

	AxisPtr = spheres.X + offset;
	// X axis
	for (uint32_t i = offset; i < end; i+=10)
	{
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
	}
	AxisPtr = spheres.Y + offset;
	// Y Axis
	for (uint32_t i = offset; i < end;   )
	{

#if DELTA_TIME
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count] * deltaTime;
#else
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
		*(AxisPtr++) += sphere_velocity[i++ + sphere_count];
#endif
	}
	AxisPtr = spheres.Y + offset;
	// X axis
	for (uint32_t i = offset; i < end; i += 10)
	{
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
	}
#if THREE_D

	AxisPtr = spheres.Z + offset;
	unsigned int arrayOffset = 2 * sphere_count;
	// Z Axis
	for (uint32_t i = offset; i < end;)
	{

#if DELTA_TIME
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset] * deltaTime;
#else
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
		*(AxisPtr++) += sphere_velocity[i++ + arrayOffset];
#endif
}
	AxisPtr = spheres.Z + offset;
	// Z axis
	for (uint32_t i = offset; i < end; i += 10)
	{
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
		if (*AxisPtr < kWallArea[0] || *AxisPtr > kWallArea[1]) *AxisPtr = -*AxisPtr;
		AxisPtr++;
	}
#endif
}




int main(int argc, char **argv)
{
	static_assert((SPHERE_COUNT / NumWorkers) % 10 == 0, "Sphere count needs to be in multiples of 10 for worker thread grouping and loop unraveling");
	static_assert(SPHERE_COUNT % 8 == 0, "Sphere count needs to be in powers of 2 for sorting optimization");
	static_assert(SPHERE_COUNT % NumWorkers == 0, "Sphere count needs to be a multiple of NumWorkers");
	SetupSimulation();

#if VISUALISER
	StartRenderer();
#endif




	float syncDelta = 0.0f;
	float secondDelta = 0.0f;
	unsigned int simulation_ticks = 0;
	while (true)
	{

		deltaTime = GetDeltaTime();
		syncDelta += deltaTime;
		secondDelta += deltaTime;


		/*StartTask(WorkerSimulateSphere, sphere_worker_groups);
		WaitForWorkers();*/
		//StartTask(WorkerQuickSort, sphere_count / 16, 16);
		StartTask(WorkerBubbleSort, sphere_count / NumWorkers, NumWorkers);
		WaitForWorkers();

		StartTask(WorkerMergeSortedArray, sphere_count / 64, 64);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerMergeSortedArray, sphere_count / 32, 32);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerMergeSortedArray, sphere_count / 16, 16);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerMergeSortedArray, sphere_count / 8, 8);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerMergeSortedArray, sphere_count / 4, 4);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerMergeSortedArray, sphere_count / 2, 2);
		WaitForWorkers();
		current_lockup_table = (current_lockup_table + 1) % 2;
		WorkerMergeSortedArray(0, 0, sphere_count);
		current_lockup_table = (current_lockup_table + 1) % 2;

		StartTask(WorkerBoundaryFinder, sphere_count / NumWorkers, NumWorkers);
		WaitForWorkers();

		for (unsigned int j = 0; j < AREA_SLICE_COUNT; ++j)
		{
			SphereBoundariesStart[j] = -1;
			SphereBoundariesEnd[j] = -1;
		}

		for (unsigned int i = 0; i < WORKER_COUNT; ++i)
		{
			for (unsigned int j = 0; j < AREA_SLICE_COUNT; ++j)
			{
				// Boundary Start
				if (SphereBoundariesStart[j] < 0)
				{
					SphereBoundariesStart[j] = WorkerAreaBoundaryChecksStart[i][j];
				}
				// Boundary End
				if (WorkerAreaBoundaryChecksEnd[i][j] > -1)
				{
					SphereBoundariesEnd[j] = WorkerAreaBoundaryChecksEnd[i][j];
				}
			} 
		}

		if (secondDelta > 1.0f)
		{
			secondDelta -= 1.0f;
			printf("Simulation UPS:%i\n", simulation_ticks);
			simulation_ticks = 0;
			GenerateNewRay();

			StartTask(FindClosestLineIntersections, sphere_count / NumWorkers, NumWorkers);
			WaitForWorkers();
			int foundSphereIndex = -1;
			float foundSphereDistance = 0.0f;

			for (unsigned int i = 0; i < NumWorkers; ++i)
			{
				if (line_sphere_index[i] >= 0)
				{
					if (foundSphereIndex < 0 || line_sphere_colision_distance[i] < foundSphereDistance)
					{
						foundSphereIndex = line_sphere_index[i];
						foundSphereDistance = line_sphere_colision_distance[i];
					}
				}
			}

			if (foundSphereIndex >= 0)
			{
				unsigned int sphereLookupIndex = SphereLookupTable[current_lockup_table][foundSphereIndex];
				sphere_hp[sphereLookupIndex] -= kLaserDamage;
				printf("Laser collided! Sphere: %.*s HP: %i\n",
					sphere_name_length, &sphere_names[sphere_name_length * sphereLookupIndex], // Name
					sphere_hp[sphereLookupIndex]);                                             // HP

#if SPHERE_DEATH
				if (sphere_hp[sphereLookupIndex] <= 0)
					printf("Sphere %.*s died!\n",
						sphere_name_length, &sphere_names[sphere_name_length * sphereLookupIndex]); // Name
#endif
			}

		}

#if VISUALISER
		bool hasRendered = syncDelta > kTargetFPS;
		if (hasRendered)
		{
			syncDelta -= kTargetFPS;


#if RESOLVE_ALPHA_SORTING

			StartTask(WorkerAlphaSort, sphere_count / NumWorkers, NumWorkers);
			WaitForWorkers();

#else
			memcpy(
				position_mapped_buffer_memory,                                           // The destination for our memory (GPU)
				SphereData,                                                              // Source for the memory (CPU-Ram)
				position_buffer_size                                                     // How much data we are transfering
			);
#endif



			{ // Only use haveWork if other thread is not
				std::unique_lock<std::mutex> lock(renderer_lock);
				can_render = true;
			}
			render_work_ready.notify_one(); // Tell worker

		}
#endif
#if SPHERE_TO_SPHERE_COLLISION
		StartTask(WorkerCollisionDetection, sphere_count / AREA_SLICE_COUNT, AREA_SLICE_COUNT);
#else
		StartTask(WorkerSimulateSphere, sphere_count / AREA_SLICE_COUNT, AREA_SLICE_COUNT);
#endif
		WaitForWorkers();
		

		simulation_ticks += 1;

#if VISUALISER
		if (hasRendered)
		{
			WaitForRender();
			PollWindow();
		}
#endif
	}

#if VISUALISER
	DestroyPipelines();

	DestroyModelBuffers();

	DestroyVulkan();
#endif
	return 0;
}