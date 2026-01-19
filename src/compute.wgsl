${commons}

struct Storage {
    index: array<f32, ${meshArraySize}>,
    position: array<f32, ${meshArraySize}>,
    normal: array<f32, ${meshArraySize}>,
    uv: array<f32, ${meshArraySize}>,
    objects: array<SceneObject, ${objectsArraySize}>,
    materials: array<SceneMaterial, ${materialsArraySize}>,
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
    p1: f32,
    p2: f32,
    p3: f32,
}

struct SceneMaterial {
    baseColor: vec4f,
    emissiveColor: vec4f,
    metallic: f32,
    roughness: f32,
    ior: f32,
    transmission: f32,
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
    object: u32,
    face: u32,
    distance: f32,
}

const maxDistance = 1e10;
const maxBounces = ${maxBounces};
const samplesPerPass = ${samplesPerPass};

var<private> testCountTriangle = 0.;
var<private> testCountAabb = 0.;

@group(0) @binding(0) var acc: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> store: Storage;

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
    color /= samplesPerPass;
    color.a = testCountTriangle / 1e3;
    // color.a = testCountAabb / 1e1;

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
    let ambientEmission = .1;
    let ambientColor = vec3f(1);

    var color = vec3f(ambientColor);
    var emission = 1.;
    var ray = rayStart;
    var ior = 1.;

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let rayCast = castRay(ray);

        if rayCast.intersection.hit {
            let object = store.objects[rayCast.object];
            let material = store.materials[u32(object.material)];

            if material.emissiveColor.a > 1 {
                emission *= material.emissiveColor.a;
                break;
            }

            let normalWorld = rayCast.normal;
            let cosIncidenceWorld = dot(ray.dir, normalWorld);
            let outsideIn = cosIncidenceWorld < 0;

            var normal = normalWorld;
            var cosIncidence = cosIncidenceWorld;
            if !outsideIn {
                normal *= -1;
                cosIncidence *= -1;
            }

            let reflection = ray.dir - 2 * cosIncidence * normal;
            let scatter = randomDir3();

            var iorFrom: f32;
            var iorTo: f32;
            if outsideIn {
                iorFrom = ior;
                iorTo = material.ior;
            } else {
                iorFrom = material.ior;
                // TODO: what if flying out into another object?
                iorTo = 1;
            }

            let nonMetalReflectance = 0.04;
            let colorDiffuse = material.baseColor.rgb;
            // TODO: colorSpecular from material
            let colorSpecular = material.baseColor.rgb;
            let reflectance = schlickFresnel(cosIncidence, iorFrom, iorTo);
            let isReflection = clamp(max(material.metallic, reflectance), nonMetalReflectance, 1) > randomf();

            var dir: vec3f;
            if isReflection {
                color *= colorSpecular;
                dir = lerp3(reflection, scatter, material.roughness);
            } else {
                let isTransmission = material.transmission > randomf();
                if isTransmission {
                    let refraction = refractionDirSnell(ray.dir, normal, cosIncidence, iorFrom, iorTo);
                    let totalInternal = refraction.w == 0;
                    if totalInternal {
                        color *= colorSpecular;
                        dir = lerp3(reflection, scatter, material.roughness);
                        ior = iorFrom;
                    } else {
                        color *= colorSpecular;
                        dir = lerp3(refraction.xyz, scatter, material.roughness);
                        ior = iorTo;
                    }
                } else {
                    color *= colorDiffuse;
                    dir = scatter;
                }
            }


            ray = Ray(rayCast.intersection.point, dir, 1 / dir);
        } else {
            emission = 0;
            break;
        }
        if bounce == maxBounces - 1 {
            emission = ambientEmission;
        }
    }

    return color * emission;
}

fn castRay(ray: Ray) -> RayCast {
    var rayCast = RayCast();
    rayCast.distance = maxDistance;
    for (var i = 0u; i < u32(store.objectCount); i++) {
        let object = store.objects[i];
        if !intersectAabb(ray, object.boundingBox) {
            continue;
        }
        let indexOffset = u32(object.indexOffset);
        let vertexOffset = u32(object.vertexOffset);
        for (var fi = 0u; fi < u32(object.indexCount / 3); fi++) {
            var triangle: array<vec3f, 3>;
            for (var v = 0u; v < 3; v++) {
                let triIndex = u32(store.index[indexOffset + 3 * fi + v]);
                let triIndexGlobal = 3 * (vertexOffset + triIndex);
                let trianglePos = vec3f(
                    store.position[triIndexGlobal],
                    store.position[triIndexGlobal + 1],
                    store.position[triIndexGlobal + 2],
                );
                triangle[v] = trianglePos;
            }
            let intersection = intersectTriangle(ray, triangle);
            if intersection.hit {
                let d = distance(intersection.point, ray.origin);
                if d < rayCast.distance {
                    var triNormals: array<vec3f, 3>;
                    for (var v = 0u; v < 3; v++) {
                        let triIndex = u32(store.index[indexOffset + 3 * fi + v]);
                        let triIndexGlobal = 3 * (vertexOffset + triIndex);
                        let vertexNormal = vec3f(
                            store.normal[triIndexGlobal],
                            store.normal[triIndexGlobal + 1],
                            store.normal[triIndexGlobal + 2],
                        );
                        triNormals[v] = vertexNormal;
                    }
                    let u = intersection.uv.x;
                    let v = intersection.uv.y;
                    let normal = normalize(
                        triNormals[0] + (triNormals[1] - triNormals[0]) * u + (triNormals[2] - triNormals[0]) * v
                    );

                    if dot(normal, ray.dir) > 0 {
                        let material = store.materials[u32(object.material)];
                        if material.transmission == 0 {
                            continue;
                        }
                    }

                    rayCast.intersection = intersection;
                    rayCast.normal = normal;
                    rayCast.object = i;
                    rayCast.face = fi;
                    rayCast.distance = d;
                }
            }
        }
    }
    return rayCast;
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
    let offsetSubpixel = random2f() * 2 - 1;
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

fn intersectAabb(ray: Ray, aabb: Aabb) -> bool {
    testCountAabb += 1;
    let t1 = (aabb.min - ray.origin) * ray.dirInv;
    let t2 = (aabb.max - ray.origin) * ray.dirInv;

    var tmin = min(t1.x, t2.x);
    tmin = max(tmin, min(t1.y, t2.y));
    tmin = max(tmin, min(t1.z, t2.z));

    var tmax = max(t1.x, t2.x);
    tmax = min(tmax, max(t1.y, t2.y));
    tmax = min(tmax, max(t1.z, t2.z));

    return tmax >= max(0, tmin);
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
