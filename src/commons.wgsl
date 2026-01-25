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

fn randomCircleSample() -> vec2f {
    let theta = randomf() * 2 * pi;
    let r = sqrt(randomf());
    return vec2f( r * cos(theta), r * sin(theta));
}

// https://math.stackexchange.com/a/1585996
fn randomDir3() -> vec3f {
    return normalize(vec3f(
        randomNormalDistribution(),
        randomNormalDistribution(),
        randomNormalDistribution(),
    ));
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn lerp2(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return a + t * (b - a);
}

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + t * (b - a);
}

fn lerp4(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return a + t * (b - a);
}

fn equirectUv(dir: vec3f) -> vec2f {
    let phi = atan2(dir.z, dir.x);
    let u = (phi / (2 * pi)) + 5;
    let theta = asin(clamp(dir.y, -1, 1));
    let v = .5 - (theta / pi);
    return vec2f(u, v);
}
