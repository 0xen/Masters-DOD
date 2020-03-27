#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable




layout(binding = 0, set = 0) buffer Settings { 
	uint sphere_count;
	float aspect_ratio; 
	float camera_z; 
}
settings;

layout(location = 0) in vec3 inPosition;

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

	mat4 cam_pos_proj = projection;
	cam_pos_proj[0][0] = 1.0f / (settings.aspect_ratio * halfTanOfFOV);
	cam_pos_proj*= camera_matrix;


	vec4 world_position = vec4(inPosition,1.0f);

	gl_Position = cam_pos_proj * world_position;

}