#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable




layout(binding = 0, set = 0) buffer Settings { 
	uint sphere_count;
	float aspect_ratio; 
	float camera_z; 
}
settings;

layout(binding = 1, set = 0) buffer SphereData { float[] p; }
sphere_data;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 outUV;
layout(location = 1) out uint outID;

void main()
{


	// FOV 65*
	const float FOV = 1.13446f;
	const float halfTanOfFOV = tan(FOV / 2);

	// Hardcoded projection matrix
	// FOV 65*
	// Min Max Depth: 0.1f - 1000.0f
	// Aspect Ratio: Calulated at runtime, element [0][0]
	const mat4 projection = mat4(
		vec4(1.04645693f,0.0f,0.0f,0.0f),
		vec4(0.0f, 1.0f / halfTanOfFOV,0.0f,0.0f),
		vec4(0.0f,0.0f,-1.00010002f,-1.00000000f),
		vec4(0.0f,0.0f,-0.100000992f,0.0f)
	);

	const mat4 camera_matrix = mat4(
		vec4(1.0f,0.0f,0.0f,0.0f),
		vec4(0.0f,1.0f,0.0f,0.0f),
		vec4(0.0f,0.0f,1.0f,0.0f),
		vec4(0.0f,0.0f,settings.camera_z,1.0f)
	);
	uint xIndex = gl_InstanceIndex;
	uint yIndex = gl_InstanceIndex + settings.sphere_count;
	uint scaleIndex = gl_InstanceIndex + (2 * settings.sphere_count);

	mat4 model_matrix = mat4(
		vec4(1.0f,0.0f,0.0f,0.0f),
		vec4(0.0f,1.0f,0.0f,0.0f),
		vec4(0.0f,0.0f,1.0f,0.0f),
		vec4(sphere_data.p[xIndex],sphere_data.p[yIndex],0.0f,1.0f)
	);
	float scale = sphere_data.p[scaleIndex];

	mat4 cam_pos_proj = projection;
	cam_pos_proj[0][0] = 1.0f / (settings.aspect_ratio * halfTanOfFOV);
	cam_pos_proj*= camera_matrix;

	vec4 model_pos = vec4(inPosition * scale,1.0f);

	vec4 world_position = model_matrix * model_pos;

	gl_Position = cam_pos_proj * world_position;

	outUV = inUV;
	outID = gl_InstanceIndex;
}