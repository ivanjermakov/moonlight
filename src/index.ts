import {
    BufferAttribute,
    BufferGeometry,
    Camera,
    Matrix4,
    Mesh,
    MeshStandardMaterial,
    PerspectiveCamera,
    Quaternion,
    Vector3
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
}

type CameraConfig = {
    camera: Camera
    matrixWorld: Matrix4
    matrixRotation: Matrix4
    sensorWidth: number
    focalLength: number
}

const materials: { [name: string]: MeshStandardMaterial } = {}
const objects: SceneObject[] = []
let camera!: CameraConfig

const workgroupSize = [8, 8]
const renderScale = 1 / 2
const computeOutputTextureSize = 2048
const computeOutputTextureFormat: GPUTextureFormat = 'rgba16float'
const meshArraySize = 8192
const objectArraySize = 128

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let formatCanvas: GPUTextureFormat
let resolution: [number, number]

let computePipeline: GPUComputePipeline
let computeOutputTexture: GPUTexture
let computeBindGroup: GPUBindGroup
let uniformBuffer: GPUBuffer

let renderPipeline: GPURenderPipeline
let renderBindGroup: GPUBindGroup
let clipVertexBuffer: GPUBuffer

let frameStart: number = 0

const wgsl = String.raw

const commons = wgsl`
struct Storage {
    index: array<f32, ${meshArraySize}>,
    position: array<f32, ${meshArraySize}>,
    normal: array<f32, ${meshArraySize}>,
    uv: array<f32, ${meshArraySize}>,
    objects: array<SceneObject, ${objectArraySize}>,
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
}
struct Camera {
    matrixWorld: mat4x4f,
    matrixRotation: mat4x4f,
    sensorWidth: f32,
    focalLength: f32,
    p1: f32,
    p2: f32,
}
struct Uniforms {
    outSize: vec2f,
    renderScale: f32,
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
            materials[o.material.name] = o.material
            const geometry = o.geometry
            if (!geometry.index) {
                console.warn('no index buffer', o)
                return
            }
            const object = {
                mesh: o,
                index: geometry.index!,
                position: geometry.attributes.position,
                normal: geometry.attributes.position,
                uv: geometry.attributes.uv,
                indexOffset,
                vertexOffset,
                matrixWorld: o.matrixWorld
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
            const quat = new Quaternion()
            const matrixRotation = new Matrix4().makeRotationFromQuaternion(o.getWorldQuaternion(quat))
            camera = {
                camera: o,
                sensorWidth: o.filmGauge,
                focalLength: o.getFocalLength(),
                matrixWorld: o.matrixWorld,
                matrixRotation
            }
            console.log(o.getWorldDirection(new Vector3()))
            console.log(new Vector3(0, 0, -1).applyMatrix4(matrixRotation))
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
    device.addEventListener('uncapturederror', event => console.error(event.error.message))
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
        console.debug('resize', [canvas.width, canvas.height], resolution)
    }
    window.addEventListener('resize', resize)
    resize()

    await initCompute()
    await initRender()

    requestAnimationFrame(loop)
    // setInterval(update)
    // await update()
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

const e = 1e-7;

struct Ray {
    origin: vec3f,
    dir: vec3f,
}

struct Intersection {
    hit: bool,
    point: vec3f,
}

@group(0) @binding(0) var out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> store: Storage;

@compute @workgroup_size(${workgroupSize.join(',')})
fn main(@builtin(global_invocation_id) gid: vec3u) {
  _ = store;
  if (gid.x >= u32(uniforms.outSize.x) || gid.y >= u32(uniforms.outSize.y)) {
      return;
  }
  let pixelPos = vec3f(gid).xy;
  let cameraRay = cameraRay(pixelPos);
  var intersectionObject = 0u;
  var intersectionDistance = 1e10;
  // for (var i = 0u; i < u32(store.objectCount); i++) {
  for (var i = 0u; i < 5; i++) {
      let object = store.objects[i];
      for (var ii = 0u; ii < u32(object.indexCount); ii++) {
          let indexOffset = u32(object.indexOffset);
          let triIndex = array<u32, 3>(
              u32(store.index[indexOffset + 3 * ii]),
              u32(store.index[indexOffset + 3 * ii + 1]),
              u32(store.index[indexOffset + 3 * ii + 2]),
          );
          var triangle: array<vec3f, 3>;
          for (var p = 0u; p < 3; p++) {
              let trianglePosLocal = vec3f(
                  store.position[3 * (u32(object.vertexOffset) + triIndex[p])],
                  store.position[3 * (u32(object.vertexOffset) + triIndex[p]) + 1],
                  store.position[3 * (u32(object.vertexOffset) + triIndex[p]) + 2],
              );
              triangle[p] = (vec4f(trianglePosLocal, 1) * object.matrixWorld).xyz;
          }
          let intersection = intersectTriangle(cameraRay, triangle);
          if intersection.hit {
              let d = distance(intersection.point, cameraRay.origin);
              if d < intersectionDistance {
                  intersectionDistance = d;
                  intersectionObject = i;
              }
          }
      }
  }
  if intersectionDistance < 1e10 {
      textureStore(out, gid.xy, vec4f(f32(intersectionObject) / 10, 0, 0, 1));
      return;
  }
  let outColor = vec4f(1, 1, 1, 1);
  // let outColor = outUv(pixelPos);
  // let outColor = outCheckerboard(pixelPos);
  textureStore(out, gid.xy, outColor);
}

fn cameraRay(pixelPos: vec2f) -> Ray {
    let aspectRatio = uniforms.outSize.x / uniforms.outSize.y;
    let sensorSize = vec2f(store.camera.sensorWidth, store.camera.sensorWidth / aspectRatio);
    let pixelPosNorm = ((pixelPos + .5) / uniforms.outSize) - .5;
    // default camera transform is -Z forward, 
    // convert from mm to m
    let startLocal = vec3f(
        pixelPosNorm.x * sensorSize.x,
        pixelPosNorm.y * sensorSize.y,
        store.camera.focalLength,
    ) / 1000;
    return Ray(
        (vec4f(startLocal, 1) * store.camera.matrixWorld).xyz,
        normalize((vec4f(startLocal, 0) * store.camera.matrixRotation).xyz),
    );
}

// adapted https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#Rust_implementation
fn intersectTriangle(ray: Ray, triangle: array<vec3f, 3>) -> Intersection {
    var intersection = Intersection(false, vec3f());
	let e1 = triangle[1] - triangle[0];
	let e2 = triangle[2] - triangle[0];

	let ray_cross_e2 = cross(ray.dir, e2);
	let det = dot(e1, ray_cross_e2);

	if det > -e && det < e {
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
	if t <= e {
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
            { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: computeOutputTextureFormat } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
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
    const objectArray: number[] = []
    for (const o of objects) {
        indexArray.set(o.index.array, o.indexOffset)
        positionArray.set(o.position.array, o.vertexOffset * 3)
        normalArray.set(o.normal.array, o.vertexOffset * 3)
        uvArray.set(o.uv.array, o.vertexOffset * 2)
        objectArray.push(...o.matrixWorld.toArray(), o.indexOffset, o.index.count, o.vertexOffset, o.position.count)
    }
    const objectsTypedArray = new Float32Array(20 * objectArraySize)
    objectsTypedArray.set(objectArray)
    const cameraArray = [
        ...camera.matrixWorld.toArray(),
        ...camera.matrixRotation.toArray(),
        camera.sensorWidth,
        // camera.focalLength,
        5,
        0,
        0
    ]
    const storageBufferArray = [
        ...indexArray,
        ...positionArray,
        ...normalArray,
        ...uvArray,
        ...objectsTypedArray,
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
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    })

    computeBindGroup = device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: computeOutputTexture.createView() },
            { binding: 1, resource: uniformBuffer },
            { binding: 2, resource: storageBuffer }
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
            { binding: 0, resource: computeOutputTexture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: uniformBuffer }
        ]
    })
}

const loop = async () => {
    await update()
    requestAnimationFrame(loop)
}

const update = async () => {
    const start = performance.now()

    compute()
    draw()
    await device.queue.onSubmittedWorkDone()

    document.getElementById('delta')!.innerText = (start - frameStart).toFixed(2).padStart(5, ' ')
    frameStart = start
}

const writeUniforms = () => {
    const uniforms = new Float32Array([...resolution, renderScale])
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

main()
