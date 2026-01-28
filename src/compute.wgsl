${commons}

${bvh}

struct Storage {
    index: array<f32, ${objectsArraySize} * ${indexSizePerMesh}>,
    position: array<f32, ${objectsArraySize} * 3 * ${vertexSizePerMesh}>,
    normal: array<f32, ${objectsArraySize} * 3 * ${vertexSizePerMesh}>,
    uv: array<f32, ${objectsArraySize} * 2 * ${vertexSizePerMesh}>,
    tangent: array<f32, ${objectsArraySize} * 4 * ${vertexSizePerMesh}>,
    bvhNode: array<BvhNode, ${bvhNodeArraySize}>,
    // array of object-space triangle indices, indexed by bvhNode
    bvhTriangle: array<f32, ${objectsArraySize} * ${indexSizePerMesh}>,
    objects: array<SceneObject, ${objectsArraySize}>,
    materials: array<SceneMaterial, ${materialsArraySize}>,
    sceneBvhNode: array<BvhNode, ${sceneBvhNodeArraySize}>,
    // array of object indices, indexed by sceneBvhNodes
    sceneBvhObject: array<f32, ${objectsArraySize}>,
    camera: Camera,
    objectCount: f32,
    p1: f32,
    p2: f32,
    p3: f32,
}

struct SceneObject {
    boundingBox: Aabb,
    indexOffset: f32,
    indexCount: f32,
    vertexOffset: f32,
    vertexCount: f32,
    material: f32,
    bvhOffset: f32,
    bvhCount: f32,
    p1: f32,
}

struct SceneMaterial {
    baseColor: vec4f,
    emissiveColor: vec4f,
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
    map: f32,
    mapNormal: f32,
    p2: f32,
    p3: f32,
}

struct Camera {
    matrixWorld: mat4x4f,
    rotation: vec4f,
    sensorWidth: f32,
    focalLength: f32,
    focus: f32,
    fstop: f32,
}

struct Ray {
    origin: vec3f,
    dir: vec3f,
    dirInv: vec3f,
}

struct Aabb {
    min: vec3f,
    // p1: f32,
    max: vec3f,
    // p2: f32,
}

struct Intersection {
    hit: bool,
    point: vec3f,
    uv: vec2f,
}

struct RayCast {
    intersection: Intersection,
    normal: vec3f,
    tangent: vec4f,
    uv: vec2f,
    object: u32,
    face: u32,
    distance: f32,
}

const maxDistance = 1e10;
const maxBounces = ${maxBounces};
const maxBouncesDiffuse = ${maxBouncesDiffuse};
const maxBouncesSpecular = ${maxBouncesSpecular};
const maxBouncesTransmission = ${maxBouncesTransmission};
const samplesPerPass = ${samplesPerPass};

var<private> testCountTriangle = 0.;
var<private> testCountAabb = 0.;
var<private> bounceCount = 0.;

@group(0) @binding(0) var acc: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> store: Storage;
@group(0) @binding(4) var mapsTexture: texture_2d_array<f32>;
@group(0) @binding(5) var envTexture: texture_2d<f32>;
@group(0) @binding(6) var textureSampler: sampler;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u32(uniforms.outSize.x) || gid.y >= u32(uniforms.outSize.y)) {
        return;
    }
    seed = (gid.x + 142467) * (gid.y + 452316) * (u32(uniforms.frame) + 264243);
    let pixelPos = vec3f(gid).xy;

    let cameraRay = cameraRay(pixelPos);
    var color = vec4f(0);
    for (var i = 0u; i < samplesPerPass; i++) {
        color += vec4f(traceRay(pixelPos, cameraRay), 0);
    }
    color.a = testCountTriangle / .5e3;
    // color.a = testCountAabb / 1e1;
    // color.a = bounceCount / maxBounces;
    color /= samplesPerPass;

    if uniforms.frame == 0 {
        textureStore(out, gid.xy, color);
        return;
    }
    let weight = 1 / uniforms.frame;
    let oldColor = textureLoad(acc, gid.xy);
    let outColor = oldColor * (1 - weight) + color * weight;
    textureStore(out, gid.xy, outColor);
}

fn traceRay(pixelPos: vec2f, rayStart: Ray) -> vec3f {
    var bouncesDiffuse = 0;
    var bouncesSpecular = 0;
    var bouncesTransmission = 0;

    let wavelength = lerp(400, 700, randomf());
    // linearly picking wavelength does not produce pure white, requires correction
    let whiteBalanceCorrection = 4.25 * vec3f(.68, .5, 1.);
    var color = wavelengthToColor(wavelength) * whiteBalanceCorrection;
    var emission = 0.;
    var ray = rayStart;

    for (var bounce = 0u; bounce < maxBounces + 1; bounce++) {
        let rayCast = castRay(ray);

        if rayCast.intersection.hit {
            let object = store.objects[rayCast.object];
            let material = store.materials[u32(object.material)];

            if material.emissiveColor.a > 0 {
                emission += material.emissiveColor.a;
                color *= material.emissiveColor.rgb;
                break;
            }

            var normalWorld = rayCast.normal;
            if (material.mapNormal > 0) {
                let tangent = normalize(rayCast.tangent.xyz);
                let bitangent = normalize(cross(tangent, normalWorld) * rayCast.tangent.w);
                let tbnMat = mat3x3f(tangent, bitangent, normalWorld);

                var normalM = textureSampleLevel(mapsTexture, textureSampler, rayCast.uv, u32(material.mapNormal), 0).rgb;
                normalM = normalM * 2 - 1;
                normalWorld = normalize(tbnMat * normalM);
            }

            let cosIncidenceWorld = dot(ray.dir, normalWorld);
            let outsideIn = dot(ray.dir, rayCast.normal) < 0;

            var normal = normalWorld;
            var cosIncidence = cosIncidenceWorld;
            var offsetDir = 1.;
            if !outsideIn {
                normal *= -1;
                cosIncidence *= -1;
                offsetDir *= -1;
            }

            let ior = dynamicIorCauchy(material.ior, wavelength);
            let iorFrom = select(ior, 1., outsideIn);
            let iorTo = select(1., ior, outsideIn);

            var colorDiffuse = material.baseColor.rgb;
            if material.map > 0 {
                colorDiffuse = textureSampleLevel(mapsTexture, textureSampler, rayCast.uv, u32(material.map), 0).rgb;
                // from srgb to linear
                // TODO: move to common fn
                // TODO: assumes all material.map textures are sRGB, which is not guaranteed by glTF
                colorDiffuse = pow(colorDiffuse, vec3f(2.4));
            }
            // TODO: colorSpecular from material
            let colorSpecular = colorDiffuse;
            let reflection = ray.dir - 2 * cosIncidence * normal;
            var scatter = randomDir3();
            if dot(scatter, normal) < 0 {
                scatter *= -1;
            }
            let isMetallic = material.metallic > randomf();
            var dir: vec3f;
            if isMetallic {
                if bouncesSpecular >= maxBouncesSpecular { break; }
                bouncesSpecular++;
                color *= colorSpecular;
                dir = lerp3(reflection, scatter, material.roughness);
            } else {
                let nonMetalReflectance = 0.04;
                let reflectance = schlickFresnel(cosIncidence, iorFrom, iorTo);
                let isReflection = max(nonMetalReflectance, reflectance) > randomf();
                if isReflection {
                    if bouncesSpecular >= maxBouncesSpecular { break; }
                    bouncesSpecular++;
                    dir = lerp3(reflection, scatter, material.roughness);
                } else {
                    let isTransmission = material.transmission > 0 && material.transmission > randomf();
                    if isTransmission {
                        let refraction = refractionDirSnell(ray.dir, normal, cosIncidence, iorFrom, iorTo);
                        let totalInternal = refraction.w == 0;
                        if totalInternal {
                            if bouncesSpecular >= maxBouncesSpecular { break; }
                            bouncesSpecular++;
                            color *= colorSpecular;
                            dir = lerp3(reflection, scatter, material.roughness);
                        } else {
                            if bouncesTransmission >= maxBouncesTransmission { break; }
                            bouncesTransmission++;
                            color *= colorSpecular;
                            dir = lerp3(refraction.xyz, scatter, material.roughness);
                            offsetDir *= -1;
                        }
                    } else {
                        if bouncesDiffuse >= maxBouncesDiffuse { break; }
                        bouncesDiffuse++;
                        color *= colorDiffuse;
                        dir = scatter;
                    }
                }
            }

            ray = Ray(rayCast.intersection.point + offsetDir * epsilon * rayCast.normal, dir, 1 / dir);
        } else {
            var envMapUv = equirectUv(-ray.dir);
            envMapUv.y *= .5;
            let raw = textureSampleLevel(envTexture, textureSampler, envMapUv, 0).rgb;
            return color * raw;
        }
        if bounce == maxBounces {
            emission = .1;
        }
        bounceCount += 1.;
    }

    return color * emission;
}

fn cameraRay(pixelPos: vec2f) -> Ray {
    let aspect = uniforms.outSize.x / uniforms.outSize.y;
    var sensorSize: vec2f;
    if aspect > uniforms.aspectRatio {
        let fitHeight = store.camera.sensorWidth / uniforms.aspectRatio;
        sensorSize = vec2f(fitHeight * aspect, fitHeight);
    } else {
        sensorSize = vec2f(store.camera.sensorWidth, store.camera.sensorWidth / aspect);
    }
    let offsetSubpixel = random2f() - .5;
    let pixelPosNorm = ((pixelPos + .5 + offsetSubpixel) / uniforms.outSize) - .5;
    let dirLocal = normalize(vec3f(
        pixelPosNorm.x * sensorSize.x,
        pixelPosNorm.y * sensorSize.y,
        -store.camera.focalLength,
    ));
    if store.camera.focus == 0 || store.camera.fstop == 0 {
        let origin = transformPoint(vec3f(), store.camera.matrixWorld);
        let dir = applyQuaternion(dirLocal, store.camera.rotation);
        return Ray(origin, dir, 1 / dir);
    } else {
        let focusPoint = dirLocal * store.camera.focus;
        let dofStrength = (1 / store.camera.fstop) * (sensorSize.x / 1000);
        let originLocal = vec3f(randomCircleSample() * dofStrength, 0);
        let dirLocal = normalize(focusPoint - originLocal);
        let origin = transformPoint(originLocal, store.camera.matrixWorld);
        let dir = applyQuaternion(dirLocal, store.camera.rotation);
        return Ray(origin, dir, 1 / dir);
    }
}

fn transformPoint(point: vec3f, mat: mat4x4f) -> vec3f {
    var v4 = vec4f(point, 1);
    v4 = mat * v4;
    if v4.w != 0 {
        return v4.xyz / v4.w;
    }
    return v4.xyz;
}

fn transformDir(dir: vec3f, mat: mat4x4f) -> vec3f {
    var v4 = vec4f(dir, 0);
    v4 = mat * v4;
    return normalize(v4.xyz);
}

fn applyQuaternion(dir: vec3f, quat: vec4f) -> vec3f {
    let t = 2 * cross(quat.xyz, dir);
    return dir + quat.w * t + cross(quat.xyz, t);
}

// adapted https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#Rust_implementation
fn intersectTriangle(ray: Ray, triangle: array<vec3f, 3>) -> Intersection {
    testCountTriangle += 1;
    var intersection = Intersection();
	let e1 = triangle[1] - triangle[0];
	let e2 = triangle[2] - triangle[0];

	let ray_cross_e2 = cross(ray.dir, e2);
	let det = dot(e1, ray_cross_e2);

	if det > -epsilon && det < epsilon {
		return intersection;
	}

	let inv_det = 1 / det;
	let s = ray.origin - triangle[0];
	let u = inv_det * dot(s, ray_cross_e2);
	if u < 0 || u > 1 {
		return intersection;
	}

	let s_cross_e1 = cross(s, e1);
	let v = inv_det * dot(ray.dir, s_cross_e1);
	if v < 0 || u + v > 1 {
		return intersection;
	}

	let t = inv_det * dot(e2, s_cross_e1);
	if t <= epsilon {
        return intersection;
    }

    intersection.hit = true;
    intersection.point = ray.origin + ray.dir * t;
    intersection.uv = vec2f(u, v);
    return intersection;
}

// distance to hit, maxDistance otherwise
fn intersectAabb(ray: Ray, aabb: Aabb) -> f32 {
    testCountAabb += 1;
    let t1 = (aabb.min - ray.origin) * ray.dirInv;
    let t2 = (aabb.max - ray.origin) * ray.dirInv;

    var tmin = min(t1.x, t2.x);
    tmin = max(tmin, min(t1.y, t2.y));
    tmin = max(tmin, min(t1.z, t2.z));

    var tmax = max(t1.x, t2.x);
    tmax = min(tmax, max(t1.y, t2.y));
    tmax = min(tmax, max(t1.z, t2.z));

    let hit = tmax >= max(0, tmin);
    return select(maxDistance, tmin, hit);
}

fn outUv(pixelPos: vec2f) -> vec4f {
    let uv = pixelPos / uniforms.outSize;
    return vec4f(uv, 0, 1);
}

fn outCheckerboard(pixelPos: vec2f) -> vec4f {
    if pixelPos.x < 1 ||
       pixelPos.y < 1 ||
       pixelPos.x > uniforms.outSize.x - 2 ||
       pixelPos.y > uniforms.outSize.y - 2 {
        return vec4f(1, 0, 0, 1);
    } else if (pixelPos.x + pixelPos.y) % 2 == 0 {
        return vec4f(0, 0, .5, 1);
    } else {
        return vec4f(1, 1, 1, 1);
    }
}

fn schlickFresnel(cosIncidence: f32, n1: f32, n2: f32) -> f32 {
    let r0 = pow((n1 - n2) / (n1 + n2), 2);
    return r0 + (1 - r0) * pow(1 - abs(cosIncidence), 5);
}

fn refractionDirSnell(incident: vec3f, normal: vec3f, cosIncidence: f32, n1: f32, n2: f32) -> vec4f {
    let eta = n1 / n2;
    let k = 1 - pow(eta, 2) * (1 - pow(cosIncidence, 2));
    if (k < 0) {
        return vec4f(0);
    } else {
        let refracted = eta * incident + (eta * -cosIncidence - sqrt(k)) * normal;
        return vec4f(refracted, 1);
    }
}

// @see https://en.wikipedia.org/wiki/Cauchy's_equation
fn dynamicIorCauchy(base: f32, wavelength: f32) -> f32 {
    let b = .01 / pow(wavelength / 1000, 2);
    return base + b;
}
