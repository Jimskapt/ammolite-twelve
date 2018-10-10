#version 450
#extension GL_ARB_separate_shader_objects : enable

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec3 fragment_colors;

vec2 positions[3] = vec2[](
    vec2(-0.50, -0.50),
    vec2(+0.50, -0.50),
    vec2(+0.00, +0.75)
);

vec3 colors[3] = vec3[](
    vec3(1.00, 1.00, 0.00),
    vec3(0.00, 1.00, 1.00),
    vec3(1.00, 0.00, 1.00)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragment_colors = colors[gl_VertexIndex];
}
