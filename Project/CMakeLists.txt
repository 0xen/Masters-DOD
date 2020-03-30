cmake_minimum_required(VERSION 2.6)

set(project_name Project)
project(${project_name})

set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG
    ${CMAKE_HOME_DIRECTORY}/Project/bin
    )
set(
    CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE
    ${CMAKE_HOME_DIRECTORY}/Project/bin
    )

set(HAVE_LIBC TRUE)
set(src
    main.cpp

)
 
set(headers

)

include_directories(../third_party/Vulkan/Vk-Helper/include)

add_executable(${project_name} ${src} ${headers})

find_package(Vulkan)



target_link_libraries(${project_name} SDL2main)
target_link_libraries(${project_name} SDL2-static)


target_link_libraries( ${project_name} debug ../../third_party/Vulkan/build/Vk-Helper/Debug/Vk-Helper)
target_link_libraries( ${project_name} optimized ../../third_party/Vulkan/build/Vk-Helper/Release/Vk-Helper)


if(Vulkan_FOUND)
	target_include_directories(${project_name} PRIVATE Vulkan::Vulkan)
	target_link_libraries(${project_name} Vulkan::Vulkan)
endif()