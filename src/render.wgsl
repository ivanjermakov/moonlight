${commons}

@group(0) @binding(0) var computeTexture: texture_2d<f32>;
@group(0) @binding(1) var computeSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

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

    let debug = false;
    if debug {
        return vec4f(color.a) / 1e3;
    } else {
        let exposure = 1.;
        var toneMapped = tmoAces(color.rgb / exposure);
        toneMapped = linearToSrgb(toneMapped);
        return vec4f(toneMapped, 1);
    }
}

fn tmoAces(x_: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let x = x_ * 0.6;
    let co = (x * (a * x + b)) / (x * (c * x + d) + e);
    return clamp(co, vec3f(0), vec3f(1));
}

fn linearToSrgb(v: vec3f) -> vec3f {
    let cutoff = step(vec3(0.0031308), v);
    let lower = v * 12.92;
    let higher = 1.055 * pow(v, vec3(1.0 / 2.4)) - 0.055;
    return mix(lower, higher, cutoff);
}
