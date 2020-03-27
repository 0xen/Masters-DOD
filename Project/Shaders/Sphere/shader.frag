#version 450

layout(binding = 2, set = 0) buffer SphereColors { 
	vec4 c[];
}
colors;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inID;

// Used with the UV coordinate to calculate where a static point 
// light would be in the scene for basic diffuse light
const vec2 LightOnSphere = vec2(1.5f,-1.3f);

// What is the min max power of the light.
// This is to stop the object being too bright or too dark
const vec2 LightPower = vec2(0.5f, 1.0f);

void main() 
{
	if(length(inUV)>0.99f)
	{
		// Output transparancy as this is byond the cecumfrance of the sphere
		outColor = vec4(0.0f,0.0f,0.0f,0.0f);
	}
	else
	{
		// Get the spheres input color
		vec3 sphere_color = vec3(colors.c[inID].rgb);
		// Calculate where the light is reletive to the spheres surface
		vec2 light_position = inUV * LightOnSphere;
		// Since the UV can be in a range of -1 - 1, we need to move both the x and y into a range of 0 - 1
		// To do this, we add them together as well as add 2, to give us a range of 0 - 4 and then multiply by 0.25
		// to give us the desired range.

		// We clamp the final range of 0 - 1 to LightPower.x - LightPower.y and multiply the sphere color by the final
		// value
		sphere_color *= clamp((light_position.x + light_position.y + 2.0f) * 0.25f, LightPower.x, LightPower.y);

		// Give the color a alpha now so it will be rendered
		outColor = vec4(sphere_color,1.0f);
	}
}