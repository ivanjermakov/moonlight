${commons}

@group(0) @binding(0) var computeTexture: texture_2d<f32>;
@group(0) @binding(1) var computeSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

const debugOverlay = ${debugOverlay};

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
};

@vertex
fn mainVertex(@location(0) position: vec2f) -> VertexOut {
    return VertexOut(vec4f(position, 0., 1.), position * .5 + .5);
}

@fragment
fn mainFragment(vout: VertexOut) -> @location(0) vec4f {
    let os = uniforms.outSize - 1;
    var uv = vout.uv * (os);
    let exactSize = floor(os);
    uv += .5 * (exactSize - uniforms.outSize);
    uv += 1;
    uv /= ${computeOutputTextureSize};

    let color = textureSample(computeTexture, computeSampler, uv);

    if debugOverlay {
        if color.a > 1 {
            return vec4f(color.a, 0, 0, 1);
        } else if color.a > .5 {
            return vec4f(color.a, color.a, 0, 1);
        } else {
            return vec4f(vec3f(color.a), 1);
        }
    } else {
        let exposure = 1.;
        var toneMapped = tfAces(color.rgb / exposure);
        toneMapped = linearToSrgb(toneMapped);
        return vec4f(toneMapped, 1);
    }
}

fn rttOdtFit(v: vec3f) -> vec3f {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn tfAces(v: vec3f) -> vec3f {
    let inputMatrix = mat3x3f(
		vec3f(0.59719, 0.07600, 0.02840),
		vec3f(0.35458, 0.90834, 0.13383),
		vec3f(0.04823, 0.01566, 0.83777)
    );
    let outputMatrix = mat3x3f(
		vec3f( 1.60475, -0.10208, -0.00327),
		vec3f(-0.53108,  1.10813, -0.07276),
		vec3f(-0.07367, -0.00605,  1.07602)
    );
    var va = inputMatrix * v;
    va = rttOdtFit(va);
    return outputMatrix * va;
}

fn linearToSrgb(v: vec3f) -> vec3f {
    let cutoff = step(vec3(0.0031308), v);
    let lower = v * 12.92;
    let higher = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
    return mix(lower, higher, cutoff);
}
