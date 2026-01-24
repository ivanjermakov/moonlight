import {
    Box3,
    BufferAttribute,
    BufferGeometry,
    Camera,
    Color,
    Matrix4,
    Mesh,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Object3D,
    PerspectiveCamera,
    Quaternion,
    Triangle
} from 'three'
import * as gltfLoader from 'three/examples/jsm/loaders/GLTFLoader.js'
import Color4 from 'three/src/renderers/common/Color4.js'
import { BvhNode, buildBvhObjects, buildBvhTris, traverseBfs, triangleByIndex } from './bvh'
import bvhWgsl from './bvh.wgsl?raw'
import commonsWgsl from './commons.wgsl?raw'
import computeWgsl from './compute.wgsl?raw'
import './index.css'
import renderWgsl from './render.wgsl?raw'
import { transformDirArray, transformPointArray } from './util'

export type SceneObject = {
    mesh: Mesh
    index: Uint16Array
    indexCount: number
    position: Float32Array
    vertexCount: number
    normal: Float32Array
    uv: Float32Array
    indexOffset: number
    vertexOffset: number
    triangles: Triangle[]
    matrixWorld: Matrix4
    material: number
    boundingBox: Box3
    bvh: BvhNode
    bvhOffset: number
    bvhCount: number
}

export type SceneMaterial = {
    material: MeshStandardMaterial
    baseColor: Color
    emissive: Color4
    metallic: number
    roughness: number
    ior: number
    transmission: number
}

export type CameraConfig = {
    camera: Camera
    matrixWorld: Matrix4
    rotation: Quaternion
    sensorWidth: number
    focalLength: number
    fstop: number
    focus: number
}

const materials: SceneMaterial[] = []
const objects: SceneObject[] = []
let camera!: CameraConfig
let sceneBvh!: BvhNode

export const renderScale = 1 / 1
export const aspectRatio = 16 / 9
export const renderHeight = 1440 as number | 'dynamic'
export const maxBounces = 8
export const maxBouncesDiffuse = 2
export const maxBouncesSpecular = 4
export const maxBouncesTransmission = 12
export const samplesPerPass = 1
/*
 * Maximum number of BVH cuts per axis to consider when splitting
 * TODO: experiment by normalizing accuracy by world size
 */
export const bvhSplitAccuracy = 5
export const sceneBvhSplitAccuracy = 512

export const timeLimit: number | undefined = 120e3
export const debugOverlay = false

export const runMode = 'busy' as 'vsync' | 'busy' | 'single'
export type SceneName =
    | 'cornell-box'
    | 'rough-metallic'
    | 'caustics'
    | 'glass'
    | 'dof'
    | 'additive-light'
    | 'primaries-sweep'
    | 'highlight-desaturation'
    | 'refraction'
    | 'refraction-foreground'
    | 'cozy-kitchen'
export const sceneName: SceneName = 'cozy-kitchen'
export const workgroupSize = [8, 8]
export const computeOutputTextureSize = 4096
export const computeOutputTextureFormat: GPUTextureFormat = 'rgba32float'

export const objectsArraySize = 1024
export const indexSizePerMesh = 8192
export const vertexSizePerMesh = 4096
export const materialsArraySize = 1024
export const sceneObjectSize = 16
export const sceneMaterialSize = 12
export const cameraSize = 24
export const bvhNodeSize = 8
export const bvhDepth = 32
export const bvhNodeArraySize = objectsArraySize * indexSizePerMesh
export const sceneBvhNodeArraySize = 2 * objectsArraySize

export const storageSize =
    // index
    objectsArraySize * indexSizePerMesh +
    // position
    objectsArraySize * 3 * vertexSizePerMesh +
    // normal
    objectsArraySize * 3 * vertexSizePerMesh +
    // uv
    objectsArraySize * 2 * vertexSizePerMesh +
    // bvhNode
    bvhNodeSize * bvhNodeArraySize +
    // bvhTriangle
    objectsArraySize * indexSizePerMesh +
    // object
    sceneObjectSize * objectsArraySize +
    // material
    sceneMaterialSize * materialsArraySize +
    // sceneBvhNode
    bvhNodeSize * sceneBvhNodeArraySize +
    // sceneBvhObject
    objectsArraySize +
    cameraSize +
    4

let device: GPUDevice
let canvas: HTMLCanvasElement
let info: HTMLElement
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
let firstFrameStart: number = 0
let frameStart: number = 0
let capture = false

const main = async (): Promise<void> => {
    info = document.getElementById('info')!
    info.innerText = 'loading'

    if (!navigator.gpu) {
        alert('WebGPU is not supported')
        return
    }
    formatCanvas = navigator.gpu.getPreferredCanvasFormat()

    try {
        device = (await initDevice())!
    } catch (e) {
        console.error(e)
        alert(`WebGPU device request failed: ${e}`)
        return
    }
    console.debug(device)

    canvas = document.getElementById('canvas') as HTMLCanvasElement
    ctx = canvas.getContext('webgpu')!
    console.debug(ctx)
    ctx.configure({ device, format: formatCanvas, alphaMode: 'premultiplied' })

    const resize = () => {
        const dpr = window.devicePixelRatio
        if (renderHeight === 'dynamic') {
            canvas.width = window.innerWidth * dpr
            canvas.height = window.innerHeight * dpr
            resolution = [canvas.width * renderScale, canvas.height * renderScale]
            canvas.style.width = '100%'
            canvas.style.height = '100%'
            frame = -1
        } else {
            resolution = [renderHeight * aspectRatio * renderScale, renderHeight * renderScale]
            canvas.width = resolution[0]
            canvas.height = resolution[1]
            canvas.style.width = `${canvas.width / renderScale / dpr}px`
            canvas.style.height = `${canvas.height / renderScale / dpr}px`
        }
        console.debug('resize', [canvas.width, canvas.height], resolution, canvas.width / canvas.height)
    }
    window.addEventListener('resize', resize)
    resize()

    window.addEventListener('keydown', async e => {
        if (e.code === 'KeyE') {
            capture = true
            if (runMode === 'single') await update()
        }
    })

    const start = performance.now()
    await initScene()
    console.debug(`init scene in ${(performance.now() - start).toFixed()}ms`)
    await initCompute()
    await initRender()

    switch (runMode) {
        case 'vsync':
            requestAnimationFrame(loop)
            return
        case 'single':
            await update()
            return
        case 'busy':
            while (true) {
                await update()
                // don't freeze browser when timeLimit is reached
                await new Promise(d => setTimeout(d))
            }
    }
}

const loop = async () => {
    await update()
    requestAnimationFrame(loop)
}

const update = async () => {
    const start = performance.now()
    const lastFrameStart = frameStart
    frameStart = start
    if (frame <= 0) {
        frame = 0
        firstFrameStart = frameStart
    }

    // TODO: don't advance time on unfocued window and vsync run mode
    const elapsed = start - firstFrameStart
    const timeOut = timeLimit !== undefined && elapsed >= timeLimit
    if (timeOut && !capture) return

    compute()
    draw()
    downloadCapture()

    if (timeOut) return

    await device.queue.onSubmittedWorkDone()

    const dt = start - lastFrameStart
    const dtps = dt / samplesPerPass
    info.innerText = [
        sceneName,
        ['dt  ', dt.toFixed(1).padStart(6, ' '), dtps.toFixed(1).padStart(5, ' ')].join(' '),
        ['fps ', (1000 / dt).toFixed(1).padStart(6, ' '), (1000 / dtps).toFixed(1).padStart(5, ' ')].join(' '),
        ['smpl', (frame * samplesPerPass).toFixed().padStart(6, ' ')].join(' '),
        [
            'time',
            `${(elapsed / 1000).toFixed().padStart(5, ' ')}s`,
            timeLimit !== undefined ? `${(timeLimit / 1000).toFixed().padStart(4, ' ')}s` : ''
        ].join(' ')
    ].join('\n')

    frame++
}

const downloadCapture = async () => {
    if (!capture) return
    capture = false
    const downCanvas = new OffscreenCanvas(canvas.width, canvas.height)
    const downCtx = downCanvas.getContext('2d')!
    downCtx.drawImage(canvas, 0, 0)
    const blob = await downCanvas.convertToBlob({ type: 'image/png' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${sceneName}-${new Date()
        .toISOString()
        .replace(/T/, '_')
        .replace(/:/g, '-')
        .replace(/\..+/, '')
        .replace(/-/, '-')}.png`
    a.click()
    URL.revokeObjectURL(a.href)
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
    return await adapter.requestDevice({
        requiredLimits: { maxBufferSize: storageSize * 4, maxStorageBufferBindingSize: storageSize * 4 }
    })
}

const initScene = async () => {
    const gltfPath = `/${sceneName}.glb`
    const gltfData = await (await fetch(gltfPath)).arrayBuffer()
    const gltf = await new gltfLoader.GLTFLoader().parseAsync(gltfData, gltfPath)
    console.log(gltf)

    const traverse = (object: Object3D) => {
        const objs: Object3D[] = [object]
        for (const child of object.children) {
            objs.push(...traverse(child))
        }
        return objs
    }
    let indexOffset = 0
    let vertexOffset = 0
    const objs = traverse(gltf.scene)
    for (const o of objs) {
        if (o instanceof Mesh && o.material instanceof MeshStandardMaterial && o.geometry instanceof BufferGeometry) {
            const material = o.material

            let materialIndex = materials.findIndex(m => m.material.name === material.name)
            if (materialIndex < 0) {
                materialIndex = materials.length
                const sceneMaterial: SceneMaterial = {
                    material,
                    baseColor: material.color,
                    emissive:
                        material.emissive.r + material.emissive.g + material.emissive.b > 0
                            ? new Color4(...material.emissive, material.emissiveIntensity)
                            : new Color4(0, 0, 0, 0),
                    metallic: material.metalness,
                    roughness: material.roughness,
                    ior: 1,
                    transmission: 0
                }
                if (material instanceof MeshPhysicalMaterial) {
                    sceneMaterial.ior = material.ior
                    sceneMaterial.transmission = material.transmission
                }
                materials.push(sceneMaterial)
            }

            const geometry = o.geometry
            const position = geometry.attributes.position as BufferAttribute
            const normal = geometry.attributes.normal as BufferAttribute
            if (!geometry.index) {
                console.warn('no index buffer', o)
                continue
            }
            if (!geometry.attributes.uv) {
                geometry.attributes.uv = new BufferAttribute(new Float32Array(position.count * 2), 2)
            }
            if (!(position.count === normal.count && position.count === geometry.attributes.uv.count)) {
                console.warn('inconsistent buffer size', o)
                continue
            }
            if (!geometry.boundingBox) {
                console.warn('no bounding box', o)
                continue
            }

            const mat = o.matrixWorld
            const object: SceneObject = {
                mesh: o,
                index: geometry.index!.array as Uint16Array,
                indexCount: geometry.index.count,
                position: transformPointArray(position.array as Float32Array, mat),
                vertexCount: position.count,
                normal: transformDirArray(normal.array as Float32Array, mat),
                uv: geometry.attributes.uv.array as Float32Array,
                indexOffset,
                vertexOffset,
                triangles: [],
                matrixWorld: mat,
                material: materialIndex,
                boundingBox: o.geometry.boundingBox!.clone().applyMatrix4(mat),
                bvh: undefined as any,
                bvhOffset: undefined as any,
                bvhCount: undefined as any
            }
            indexOffset += object.indexCount
            vertexOffset += object.vertexCount
            object.triangles = new Array(object.indexCount / 3).fill(0).map((_, i) => triangleByIndex(object, i))
            object.bvh = buildBvhTris(object, bvhSplitAccuracy)
            objects.push(object)
        }
        if (o instanceof PerspectiveCamera) {
            camera = {
                camera: o,
                sensorWidth: o.filmGauge,
                focalLength: o.getFocalLength(),
                matrixWorld: o.matrixWorld,
                rotation: o.quaternion,
                fstop: o.userData.aperture_fstop ?? 0,
                focus: o.userData.focus_distance ?? 0
            }
        }
    }
    if (indexOffset > objectsArraySize * indexSizePerMesh) throw Error('overflow')
    if (vertexOffset > objectsArraySize * 3 * vertexSizePerMesh) throw Error('overflow')
    sceneBvh = buildBvhObjects(objects, sceneBvhSplitAccuracy)

    if (objects.length === 0) throw Error('no objects')
    console.debug(objects)
    console.debug(materials)
    console.debug(camera)
    console.debug(objects.map(o => o.bvh))
    console.debug(sceneBvh)
}

const initCompute = async () => {
    const computeModule = device.createShaderModule({
        code: applyTemplate(computeWgsl, {
            commons,
            bvh,
            indexSizePerMesh,
            vertexSizePerMesh,
            objectsArraySize,
            materialsArraySize,
            bvhNodeArraySize,
            sceneBvhNodeArraySize,
            bvhDepth,
            maxBounces,
            maxBouncesDiffuse,
            maxBouncesSpecular,
            maxBouncesTransmission,
            samplesPerPass,
            workgroupSize: workgroupSize.join(',')
        })
    })

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

    const storage = new Float32Array(storageSize)
    let storageOffset = 0

    for (const o of objects) {
        storage.set(o.index, storageOffset + o.indexOffset)
    }
    storageOffset += objectsArraySize * indexSizePerMesh

    for (const o of objects) {
        storage.set(o.position, storageOffset + 3 * o.vertexOffset)
    }
    storageOffset += objectsArraySize * 3 * vertexSizePerMesh

    for (const o of objects) {
        storage.set(o.normal, storageOffset + 3 * o.vertexOffset)
    }
    storageOffset += objectsArraySize * 3 * vertexSizePerMesh

    for (const o of objects) {
        storage.set(o.uv, storageOffset + 2 * o.vertexOffset)
    }
    storageOffset += objectsArraySize * 2 * vertexSizePerMesh

    const bvhNodeArray: number[] = []
    const bvhTriangleArray: number[] = []
    for (const o of objects) {
        const bvhNodes = traverseBfs(o.bvh)
        o.bvhOffset = bvhNodeArray.length / bvhNodeSize
        o.bvhCount = bvhNodes.length
        for (const node of bvhNodes) {
            if (node.type === 'node') {
                bvhNodeArray.push(
                    ...node.box.min.toArray(),
                    o.bvhOffset + bvhNodes.indexOf(node.left),
                    ...node.box.max.toArray(),
                    0
                )
            } else {
                bvhNodeArray.push(
                    ...node.box.min.toArray(),
                    bvhTriangleArray.length,
                    ...node.box.max.toArray(),
                    node.index.length
                )
                bvhTriangleArray.push(...node.index)
            }
        }
        // const leafTris = bvhNodes
        //     .filter(l => l.type === 'leaf')
        //     .map(l => l.index.length)
        //     .toSorted((a, b) => a - b)
        // console.debug('bvh', o.mesh.name, leafTris, {
        //     min: leafTris.reduce((a, b) => (b < a ? b : a), Number.POSITIVE_INFINITY),
        //     max: leafTris.reduce((a, b) => (b > a ? b : a), 0),
        //     mean: leafTris[Math.floor(leafTris.length / 2)],
        //     avg: leafTris.reduce((a, b) => a + b, 0) / leafTris.length
        // })
    }
    if (bvhNodeArray.length > bvhNodeSize * bvhNodeArraySize) throw Error('storage overflow')
    storage.set(bvhNodeArray, storageOffset)
    storageOffset += bvhNodeSize * bvhNodeArraySize
    if (bvhTriangleArray.length > objectsArraySize * indexSizePerMesh) throw Error('storage overflow')
    storage.set(bvhTriangleArray, storageOffset)
    storageOffset += objectsArraySize * indexSizePerMesh

    let objectOffset = 0
    for (const o of objects) {
        storage.set(
            [
                ...[...o.boundingBox.min.toArray(), 0, ...o.boundingBox.max.toArray(), 0],
                o.indexOffset,
                o.indexCount,
                o.vertexOffset,
                o.vertexCount,
                o.material,
                o.bvhOffset,
                o.bvhCount,
                0
            ],
            storageOffset + objectOffset
        )
        objectOffset += sceneObjectSize
    }
    if (objectOffset > sceneObjectSize * objectsArraySize) throw Error('storage overflow')
    storageOffset += sceneObjectSize * objectsArraySize

    let materialOffset = 0
    for (const m of materials) {
        storage.set(
            [...m.baseColor.toArray(), 1, ...m.emissive, m.emissive.a, m.metallic, m.roughness, m.ior, m.transmission],
            storageOffset + materialOffset
        )
        materialOffset += sceneMaterialSize
    }
    if (materialOffset > sceneMaterialSize * materialsArraySize) throw Error('storage overflow')
    storageOffset += sceneMaterialSize * materialsArraySize

    const sceneBvhNodeArray: number[] = []
    const sceneBvhObjectArray: number[] = []
    const sceneBvhNodes = traverseBfs(sceneBvh)
    for (let i = 0; i < sceneBvhNodes.length; i++) {
        const node = sceneBvhNodes[i]
        if (node.type === 'node') {
            sceneBvhNodeArray.push(
                ...node.box.min.toArray(),
                sceneBvhNodes.indexOf(node.left),
                ...node.box.max.toArray(),
                0
            )
        } else {
            sceneBvhNodeArray.push(
                ...node.box.min.toArray(),
                sceneBvhObjectArray.length,
                ...node.box.max.toArray(),
                node.index.length
            )
            sceneBvhObjectArray.push(...node.index)
        }
    }
    // const leafObjects = sceneBvhNodes
    //     .filter(l => l.type === 'leaf')
    //     .map(l => l.index.length)
    //     .toSorted((a, b) => a - b)
    // console.debug('scene bvh', {
    //     min: leafObjects.reduce((a, b) => (b < a ? b : a), Number.POSITIVE_INFINITY),
    //     max: leafObjects.reduce((a, b) => (b > a ? b : a), 0),
    //     mean: leafObjects[Math.floor(leafObjects.length / 2)],
    //     avg: leafObjects.reduce((a, b) => a + b, 0) / leafObjects.length
    // })
    if (sceneBvhNodeArray.length > bvhNodeSize * sceneBvhNodeArraySize) throw Error('storage overflow')
    storage.set(sceneBvhNodeArray, storageOffset)
    storageOffset += bvhNodeSize * sceneBvhNodeArraySize
    if (sceneBvhObjectArray.length > objectsArraySize) throw Error('storage overflow')
    storage.set(sceneBvhObjectArray, storageOffset)
    storageOffset += objectsArraySize

    const cameraArray = [
        ...camera.matrixWorld.toArray(),
        ...camera.rotation.toArray(),
        camera.sensorWidth,
        camera.focalLength,
        camera.focus,
        camera.fstop
    ]
    storage.set(cameraArray, storageOffset)
    storageOffset += cameraSize

    storage.set([objects.length, 0, 0, 0], storageOffset)
    storageOffset += 4

    if (storageOffset !== storageSize) throw Error(`storage size mismatch, ${storageOffset} != ${storageSize}`)

    const storageBuffer = device.createBuffer({
        size: storageSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    })
    device.queue.writeBuffer(storageBuffer, 0, storage)

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
        code: applyTemplate(renderWgsl, { commons, debugOverlay, computeOutputTextureSize })
    })

    const layout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
        ]
    })
    renderPipeline = await device.createRenderPipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
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

const applyTemplate = (code: string, variables: Record<string, string | number | boolean>): string => {
    let applied = code
    for (const [name, variable] of Object.entries(variables)) {
        applied = applied.replaceAll(`\${${name}}`, variable.toString())
    }
    return applied
}

const commons = applyTemplate(commonsWgsl, {})
const bvh = applyTemplate(bvhWgsl, {})

main()
