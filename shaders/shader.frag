#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragment_colors;

layout(location = 0) out vec4 output_colors;

void main() {
    output_colors = vec4(fragment_colors, 1.0);
}
