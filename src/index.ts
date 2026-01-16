import {
    BufferAttribute,
    BufferGeometry,
    Camera,
    Color,
    Matrix4,
    Mesh,
    MeshStandardMaterial,
    PerspectiveCamera,
    Quaternion
} from 'three'
import * as gltfLoader from 'three/examples/jsm/loaders/GLTFLoader.js'
import './index.css'

type SceneObject = {
    mesh: Mesh
    index: BufferAttribute
    position: BufferAttribute
    normal: BufferAttribute
    uv: BufferAttribute
    indexOffset: number
    vertexOffset: number
    matrixWorld: Matrix4
    material: number
}

type SceneMaterial = {
    material: MeshStandardMaterial
    baseColor: Color
    emissive: Color
    metallic: number
    roughness: number
}

type CameraConfig = {
    camera: Camera
    matrixWorld: Matrix4
    rotation: Quaternion
    sensorWidth: number
    focalLength: number
}

const materials: SceneMaterial[] = []
const objects: SceneObject[] = []
let camera!: CameraConfig

const renderScale = 1 / 2
const aspectRatio = 16 / 9
const maxBounces = 8
const workgroupSize = [8, 8]
const computeOutputTextureSize = 4096
const computeOutputTextureFormat: GPUTextureFormat = 'rgba16float'
const meshArraySize = 8192
const objectsArraySize = 128
const materialsArraySize = 32
const sceneObjectSize = 24
const sceneMaterialSize = 12
type RunMode = 'vsync' | 'busy' | 'single'
const runMode = 'vsync' as RunMode

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let formatCanvas: GPUTextureFormat
let resolution: [number, number]

let computePipeline: GPUComputePipeline
let computeAccTexture: GPUTexture
let computeOutputTexture: GPUTexture
let computeBindGroup: GPUBindGroup
let uniformBuffer: GPUBuffer

let renderPipeline: GPURenderPipeline
let renderBindGroup: GPUBindGroup
let clipVertexBuffer: GPUBuffer

let frame: number = 0
let frameStart: number = 0
let capture = false

const wgsl = String.raw

const commons = wgsl`
const pi = 3.141592653589793;
const epsilon = 1e-5;

var<private> seed = 0u;

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
    matrixWorld: mat4x4f,
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
    p1: f32,
    p2: f32,
}
struct Camera {
    matrixWorld: mat4x4f,
    rotation: vec4f,
    sensorWidth: f32,
    focalLength: f32,
    p1: f32,
    p2: f32,
}
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
`

const main = async (): Promise<void> => {
    const gltfPath = '/scene.glb'
    const gltfData = await (await fetch(gltfPath)).arrayBuffer()
    const gltf = await new gltfLoader.GLTFLoader().parseAsync(gltfData, gltfPath)
    let indexOffset = 0
    let vertexOffset = 0
    gltf.scene.traverse(o => {
        if (o instanceof Mesh && o.material instanceof MeshStandardMaterial && o.geometry instanceof BufferGeometry) {
            const material = o.material
            let materialIndex = materials.findIndex(m => m.material.name === material.name)
            if (materialIndex < 0) {
                materialIndex = materials.length
                materials.push({
                    material,
                    baseColor: material.color,
                    emissive: material.emissive,
                    metallic: material.metalness,
                    roughness: material.roughness
                })
            }
            const geometry = o.geometry
            if (!geometry.index) {
                console.warn('no index buffer', o)
                return
            }
            const object: SceneObject = {
                mesh: o,
                index: geometry.index!,
                position: geometry.attributes.position,
                normal: geometry.attributes.normal,
                uv: geometry.attributes.uv,
                indexOffset,
                vertexOffset,
                matrixWorld: o.matrixWorld,
                material: materialIndex
            }
            if (!(object.position.count === object.normal.count && object.position.count === object.uv.count)) {
                console.warn('inconsistent buffer size', object)
                return
            }
            indexOffset += object.index.count
            vertexOffset += object.position.count
            objects.push(object)
        }
        if (o instanceof PerspectiveCamera) {
            camera = {
                camera: o,
                sensorWidth: o.filmGauge,
                focalLength: o.getFocalLength(),
                matrixWorld: o.matrixWorld,
                rotation: o.quaternion
            }
        }
    })
    console.debug(objects)
    console.debug(materials)
    console.debug(camera)

    if (!navigator.gpu) {
        alert('WebGPU is not supported')
        return
    }
    formatCanvas = navigator.gpu.getPreferredCanvasFormat()

    device = (await initDevice())!
    if (!device) {
        alert('no WebGPU device')
        return
    }
    console.debug(device)

    canvas = document.getElementById('canvas') as HTMLCanvasElement
    ctx = canvas.getContext('webgpu')!
    console.debug(ctx)
    ctx.configure({ device, format: formatCanvas, alphaMode: 'premultiplied' })

    const resize = () => {
        const dpr = window.devicePixelRatio
        canvas.width = window.innerWidth * dpr
        canvas.height = window.innerHeight * dpr
        resolution = [canvas.width * renderScale, canvas.height * renderScale]
        frame = 0
        console.debug('resize', [canvas.width, canvas.height], resolution)
    }
    window.addEventListener('resize', resize)
    resize()

    window.addEventListener('keydown', async e => {
        if (e.code === 'KeyE') {
            capture = true
            if (runMode === 'single') await update()
        }
    })

    await initCompute()
    await initRender()

    switch (runMode as RunMode) {
        case 'vsync':
            requestAnimationFrame(loop)
            return
        case 'single':
            await update()
            return
        case 'busy':
            while (true) {
                await update()
            }
    }
}

const loop = async () => {
    await update()
    requestAnimationFrame(loop)
}

const update = async () => {
    const start = performance.now()

    compute()
    draw()

    if (capture) {
        capture = false
        const downCanvas = new OffscreenCanvas(canvas.width, canvas.height)
        const downCtx = downCanvas.getContext('2d')!
        downCtx.drawImage(canvas, 0, 0)
        const blob = await downCanvas.convertToBlob({ type: 'image/png' })
        const a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = `render-${new Date()
            .toISOString()
            .replace(/T/, '_')
            .replace(/:/g, '-')
            .replace(/\..+/, '')
            .replace(/-/, '-')}.png`
        a.click()
        URL.revokeObjectURL(a.href)
    }

    await device.queue.onSubmittedWorkDone()

    document.getElementById('delta')!.innerText = (start - frameStart).toFixed(2).padStart(5, ' ')
    frameStart = start
    frame++
}

const writeUniforms = () => {
    const uniforms = new Float32Array([...resolution, renderScale, frame, aspectRatio])
    device.queue.writeBuffer(uniformBuffer, 0, uniforms)
}

const compute = () => {
    const commandEncoder = device.createCommandEncoder()
    const pass = commandEncoder.beginComputePass()
    pass.setPipeline(computePipeline)
    pass.setBindGroup(0, computeBindGroup)
    writeUniforms()
    pass.dispatchWorkgroups(Math.ceil(resolution[0] / workgroupSize[0]), Math.ceil(resolution[1] / workgroupSize[1]))
    pass.end()

    commandEncoder.copyTextureToTexture(
        { texture: computeOutputTexture },
        { texture: computeAccTexture },
        { width: computeOutputTextureSize, height: computeOutputTextureSize }
    )

    device.queue.submit([commandEncoder.finish()])
}

const draw = () => {
    const commandEncoder = device.createCommandEncoder()
    const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [
            {
                loadOp: 'clear',
                storeOp: 'store',
                view: ctx.getCurrentTexture().createView()
            }
        ]
    })
    passEncoder.setPipeline(renderPipeline)
    passEncoder.setVertexBuffer(0, clipVertexBuffer)
    passEncoder.setBindGroup(0, renderBindGroup)
    writeUniforms()
    passEncoder.draw(6)
    passEncoder.end()
    device.queue.submit([commandEncoder.finish()])
}

const initDevice = async (): Promise<GPUDevice | undefined> => {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return undefined
    return await adapter.requestDevice()
}

const initCompute = async () => {
    const computeModule = device.createShaderModule({
        code: wgsl`
${commons}

const maxDistance = 1e10;
const maxBounces = ${maxBounces};

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

struct Intersection {
    hit: bool,
    point: vec3f,
}

struct RayCast {
    intersection: Intersection,
    object: u32,
    face: u32,
    distance: f32,
}

@group(0) @binding(0) var acc: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var<storage, read> store: Storage;

@compute @workgroup_size(${workgroupSize.join(',')})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= u32(uniforms.outSize.x) || gid.y >= u32(uniforms.outSize.y)) {
        return;
    }
    seed = (gid.x + 142467) * (gid.y + 452316) * (u32(uniforms.frame) + 264243);
    let pixelPos = vec3f(gid).xy;

    let cameraRay = cameraRay(pixelPos);
    let color = traceRay(pixelPos, cameraRay);
    // let color = random3f();

    let weight = 1 / (uniforms.frame + 1);
    let oldColor = textureLoad(acc, gid.xy).rgb;
    let outColor = oldColor * (1 - weight) + color * weight;
    textureStore(out, gid.xy, vec4f(outColor, 1));
}

fn traceRay(pixelPos: vec2f, rayStart: Ray) -> vec3f {
    let ambientEmission = .1;
    let ambientColor = vec3f(1);

    var light = vec3f(ambientColor);
    var emission = ambientEmission;
    var ray = rayStart;

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let rayCast = castRay(ray);

        if rayCast.intersection.hit {
            let object = store.objects[rayCast.object];
            let material = store.materials[u32(object.material)];

            if material.emissiveColor.a > 1 {
                emission += material.emissiveColor.a;
                break;
            }

            light *= material.baseColor.rgb;

            // TODO: smooth shading
            let indexOffset = u32(object.indexOffset);
            let vertexOffset = u32(object.vertexOffset);
            var normalLocal = vec3f();
            for (var v = 0u; v < 3; v++) {
                let triIndex = u32(store.index[indexOffset + 3 * rayCast.face + v]);
                let triIndexGlobal = 3 * (vertexOffset + triIndex);
                let vertexNormal = vec3f(
                    store.normal[triIndexGlobal],
                    store.normal[triIndexGlobal + 1],
                    store.normal[triIndexGlobal + 2],
                );
                normalLocal += vertexNormal;
            }
            normalLocal = normalize(normalLocal);
            let normal = transformDir(normalLocal, object.matrixWorld);

            let reflection = ray.dir - 2 * dot(ray.dir, normal) * normal;
            let roughness = material.roughness;
            // let roughness = 1.;
            var scatter = randomDirection();
            if dot(normal, scatter) < 0 {
                scatter *= -1;
            }
            let dir = normalize((roughness * scatter) + ((1 - roughness) * reflection));

            let offset = normal * 0.00;
            ray = Ray(rayCast.intersection.point + offset, dir);
        } else {
            emission = 0;
            break;
        }
    }

    return light * emission;
}

fn castRay(ray: Ray) -> RayCast {
    var rayCast = RayCast(Intersection(), 0u, 0u, maxDistance);
    for (var i = 0u; i < u32(store.objectCount); i++) {
        let object = store.objects[i];
        let indexOffset = u32(object.indexOffset);
        let vertexOffset = u32(object.vertexOffset);
        for (var fi = 0u; fi < u32(object.indexCount / 3); fi++) {
            var triangle: array<vec3f, 3>;
            for (var v = 0u; v < 3; v++) {
                let triIndex = u32(store.index[indexOffset + 3 * fi + v]);
                let triIndexGlobal = 3 * (vertexOffset + triIndex);
                let trianglePosLocal = vec3f(
                    store.position[triIndexGlobal],
                    store.position[triIndexGlobal + 1],
                    store.position[triIndexGlobal + 2],
                );
                triangle[v] = transformPoint(trianglePosLocal, object.matrixWorld);
            }
            let intersection = intersectTriangle(ray, triangle);
            if intersection.hit {
                let d = distance(intersection.point, ray.origin);
                if d < rayCast.distance {
                    rayCast.intersection = intersection;
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
    let focalPosWorld = transformPoint(vec3f(), store.camera.matrixWorld);
    let pixelPosNorm = ((pixelPos + .5) / uniforms.outSize) - .5;
    let dirLocal = normalize(vec3f(
        pixelPosNorm.x * sensorSize.x,
        pixelPosNorm.y * sensorSize.y,
        -store.camera.focalLength,
    ));
    let dir = applyQuaternion(dirLocal, store.camera.rotation);
    return Ray(
        // convert from mm to m
        focalPosWorld,
        dir,
    );
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
    return v4.xyz;
}

fn applyQuaternion(dir: vec3f, quat: vec4f) -> vec3f {
    let t = 2 * cross(quat.xyz, dir);
    return dir + quat.w * t + cross(quat.xyz, t);
}

// adapted https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#Rust_implementation
fn intersectTriangle(ray: Ray, triangle: array<vec3f, 3>) -> Intersection {
    var intersection = Intersection(false, vec3f());
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
    return intersection;
}

fn outUv(pixelPos: vec2f) -> vec4f {
    let uv = pixelPos / uniforms.outSize;
    return vec4f(uv, 0., 1.);
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
`
    })

    // needed because rgba16float is not the default choice for storage textures
    const layout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: { access: 'read-only', format: computeOutputTextureFormat }
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: computeOutputTextureFormat } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
        ]
    })

    computePipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: computeModule, entryPoint: 'main' }
    })

    uniformBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    })

    const indexArray = new Float32Array(meshArraySize)
    const positionArray = new Float32Array(meshArraySize)
    const normalArray = new Float32Array(meshArraySize)
    const uvArray = new Float32Array(meshArraySize)
    const objectsArray: number[] = []
    for (const o of objects) {
        indexArray.set(o.index.array, o.indexOffset)
        positionArray.set(o.position.array, o.vertexOffset * 3)
        normalArray.set(o.normal.array, o.vertexOffset * 3)
        uvArray.set(o.uv.array, o.vertexOffset * 2)
        objectsArray.push(
            ...o.matrixWorld.toArray(),
            o.indexOffset,
            o.index.count,
            o.vertexOffset,
            o.position.count,
            o.material,
            0,
            0,
            0
        )
    }
    const objectsTypedArray = new Float32Array(sceneObjectSize * objectsArraySize)
    objectsTypedArray.set(objectsArray)

    const materialsArray: number[] = []
    for (const m of materials) {
        materialsArray.push(
            ...m.baseColor.toArray(),
            1,
            ...m.emissive,
            m.material.emissiveIntensity,
            m.metallic,
            m.roughness,
            0,
            0
        )
    }
    const materialsTypedArray = new Float32Array(sceneMaterialSize * materialsArraySize)
    materialsTypedArray.set(materialsArray)

    const cameraArray = [
        ...camera.matrixWorld.toArray(),
        ...camera.rotation.toArray(),
        camera.sensorWidth,
        camera.focalLength,
        0,
        0
    ]
    const storageBufferArray = [
        ...indexArray,
        ...positionArray,
        ...normalArray,
        ...uvArray,
        ...objectsTypedArray,
        ...materialsTypedArray,
        ...cameraArray,
        objects.length,
        0,
        0,
        0
    ]
    const storageBuffer = device.createBuffer({
        size: storageBufferArray.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })
    device.queue.writeBuffer(storageBuffer, 0, new Float32Array(storageBufferArray))

    computeOutputTexture = device.createTexture({
        size: [computeOutputTextureSize, computeOutputTextureSize, 1],
        format: computeOutputTextureFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    })

    computeAccTexture = device.createTexture({
        size: [computeOutputTextureSize, computeOutputTextureSize, 1],
        format: computeOutputTextureFormat,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    })

    computeBindGroup = device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: computeAccTexture.createView() },
            { binding: 1, resource: computeOutputTexture.createView() },
            { binding: 2, resource: uniformBuffer },
            { binding: 3, resource: storageBuffer }
        ]
    })
}

const initRender = async () => {
    // biome-ignore format:
    const clipPlane = new Float32Array([
         -1, -1,
          1,  1,
         -1,  1,
         -1, -1,
          1, -1,
          1,  1,
    ])

    clipVertexBuffer = device.createBuffer({
        size: clipPlane.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    })

    device.queue.writeBuffer(clipVertexBuffer, 0, clipPlane)

    const renderModule = device.createShaderModule({
        code: wgsl`
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
    return textureSample(computeTexture, computeSampler, uv);
}
`
    })

    renderPipeline = await device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: {
            module: renderModule,
            entryPoint: 'mainVertex',
            buffers: [
                {
                    attributes: [
                        {
                            // position
                            shaderLocation: 0,
                            offset: 0,
                            format: 'float32x2'
                        }
                    ],
                    arrayStride: 8,
                    stepMode: 'vertex'
                }
            ]
        },
        fragment: {
            module: renderModule,
            entryPoint: 'mainFragment',
            targets: [{ format: formatCanvas }]
        },
        primitive: { topology: 'triangle-list' }
    })

    const sampler = device.createSampler({
        magFilter: 'nearest',
        minFilter: 'nearest'
    })
    renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: computeAccTexture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: uniformBuffer }
        ]
    })
}

main()
