const pi = 3.141592653589793;
const epsilon = 1e-5;

var<private> seed = 0u;

struct Uniforms {
    outSize: vec2f,
    renderScale: f32,
    frame: f32,
    aspectRatio: f32,
}

fn random() -> u32 {
    seed = seed * 747796405 + 2891336453;
    var result = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    result = (result >> 22) ^ result;
    return result;
}

fn randomf() -> f32 {
    return f32(random()) / 4294967295;
}

fn random2f() -> vec2f {
    return vec2f(randomf(), randomf());
}

fn random3f() -> vec3f {
    return vec3f(randomf(), randomf(), randomf());
}

// https://stackoverflow.com/a/6178290
fn randomNormalDistribution() -> f32 {
    let theta = 2 * pi * randomf();
    let rho = sqrt(-2 * log(randomf()));
    return rho * cos(theta);
}

// https://math.stackexchange.com/a/1585996
fn randomDirection() -> vec3f {
    return normalize(vec3f(
        randomNormalDistribution(),
        randomNormalDistribution(),
        randomNormalDistribution(),
    ));
}
