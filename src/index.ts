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
    PerspectiveCamera,
    Quaternion
} from 'three'
import * as gltfLoader from 'three/examples/jsm/loaders/GLTFLoader.js'
import { BvhNode, buildBvh, traverseBfs } from './bvh'
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
    positionCount: number
    normal: Float32Array
    uv: Float32Array
    indexOffset: number
    vertexOffset: number
    matrixWorld: Matrix4
    material: number
    boundingBox: Box3
    bvh: BvhNode
}

export type SceneMaterial = {
    material: MeshStandardMaterial
    baseColor: Color
    emissive: Color
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

export const renderScale = 1 / 1
export const aspectRatio = 16 / 9
export const maxBounces = 4
export const samplesPerPass = 1
export const timeLimit: number | undefined = 10e3

export const workgroupSize = [8, 8]
export const computeOutputTextureSize = 4096
export const computeOutputTextureFormat: GPUTextureFormat = 'rgba32float'
export const objectsArraySize = 128
export const meshArraySize = objectsArraySize * 512
export const materialsArraySize = 32
export const sceneObjectSize = 16
export const sceneMaterialSize = 12
export const bvhNodeSize = 8
export const bvhDepth = 32
export const bvhNodeArraySize = objectsArraySize * 256
export const bvhSplitAccuracy = 1024
export type RunMode = 'vsync' | 'busy' | 'single'
export const runMode = 'vsync' as RunMode
export type SceneName = 'cornell-box' | 'rough-metallic' | 'caustics' | 'glass' | 'dof'
export const sceneName = 'cornell-box' as SceneName

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

    const gltfPath = `/${sceneName}.glb`
    const gltfData = await (await fetch(gltfPath)).arrayBuffer()
    const gltf = await new gltfLoader.GLTFLoader().parseAsync(gltfData, gltfPath)
    let indexOffset = 0
    let vertexOffset = 0
    gltf.scene.traverse(o => {
        if (o instanceof Mesh && o.material instanceof MeshStandardMaterial && o.geometry instanceof BufferGeometry) {
            const material = o.material
            const geometry = o.geometry
            const position = geometry.attributes.position as BufferAttribute
            const normal = geometry.attributes.normal as BufferAttribute
            const uv = geometry.attributes.uv as BufferAttribute
            let materialIndex = materials.findIndex(m => m.material.name === material.name)
            if (materialIndex < 0) {
                materialIndex = materials.length
                const sceneMaterial: SceneMaterial = {
                    material,
                    baseColor: material.color,
                    emissive: material.emissive,
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
            if (!geometry.index) {
                console.warn('no index buffer', o)
                return
            }
            if (!geometry.boundingBox) {
                console.warn('no bounding box', o)
                return
            }
            if (!(position.count === normal.count && position.count === uv.count)) {
                console.warn('inconsistent buffer size', o)
                return
            }
            const object: SceneObject = {
                mesh: o,
                index: geometry.index!.array as Uint16Array,
                indexCount: geometry.index.count,
                position: transformPointArray(position.array as Float32Array, o.matrixWorld),
                positionCount: position.count,
                normal: transformDirArray(normal.array as Float32Array, o.matrixWorld),
                uv: uv.array as Float32Array,
                indexOffset,
                vertexOffset,
                matrixWorld: o.matrixWorld,
                material: materialIndex,
                boundingBox: o.geometry.boundingBox!.clone().applyMatrix4(o.matrixWorld),
                bvh: undefined as any
            }
            object.bvh = buildBvh(object)
            indexOffset += object.indexCount
            vertexOffset += object.positionCount
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
    })
    console.debug(objects)
    console.debug(materials)
    console.debug(camera)
    console.debug(objects.map(o => o.bvh))

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
        frame = -1
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
    const lastFrameStart = frameStart
    frameStart = start
    if (frame <= 0) {
        frame = 0
        firstFrameStart = frameStart
    }

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
    return await adapter.requestDevice()
}

const initCompute = async () => {
    const computeModule = device.createShaderModule({
        code: applyTemplate(computeWgsl, {
            commons,
            bvh,
            meshArraySize,
            objectsArraySize,
            materialsArraySize,
            bvhNodeArraySize,
            bvhDepth,
            maxBounces,
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

    const indexArray = new Float32Array(meshArraySize)
    const positionArray = new Float32Array(meshArraySize)
    const normalArray = new Float32Array(meshArraySize)
    const uvArray = new Float32Array(meshArraySize)
    const objectsArray: number[] = []
    const bvhNodeArray: number[] = []
    const bvhTriangleArray: number[] = []
    for (const o of objects) {
        indexArray.set(o.index, o.indexOffset)
        positionArray.set(o.position, o.vertexOffset * 3)
        normalArray.set(o.normal, o.vertexOffset * 3)
        uvArray.set(o.uv, o.vertexOffset * 2)

        const bvhNodeOffset = bvhNodeArray.length / bvhNodeSize
        const bvhNodes = traverseBfs(o.bvh)
        for (let i = 0; i < bvhNodes.length; i++) {
            const bvhNode = bvhNodes[i]
            bvhNodeArray.push(
                ...bvhNode.box.min.toArray(),
                bvhNode.type === 'leaf' ? bvhTriangleArray.length : bvhNodeOffset + bvhNodes.indexOf(bvhNode.left),
                ...bvhNode.box.max.toArray(),
                bvhNode.type === 'leaf' ? bvhNode.triangleIdxs.length : 0
            )
            if (bvhNode.type === 'leaf') {
                bvhTriangleArray.push(...bvhNode.triangleIdxs)
            }
        }

        // const leafTris = bvhNodes.filter(l => l.type === 'leaf').map(l => l.triangles.length).toSorted((a, b) => a - b)
        // console.debug('bvh', o.mesh.name, bvhNodes, {
        //     min: leafTris.reduce((a, b) => (b < a ? b : a), Number.POSITIVE_INFINITY),
        //     max: leafTris.reduce((a, b) => (b > a ? b : a), 0),
        //     mean: leafTris[Math.floor(leafTris.length / 2)],
        //     avg: leafTris.reduce((a, b) => a + b, 0) / leafTris.length
        // })

        objectsArray.push(
            ...[...o.boundingBox.min.toArray(), 0, ...o.boundingBox.max.toArray(), 0],
            o.indexOffset,
            o.indexCount,
            o.vertexOffset,
            o.positionCount,
            o.material,
            bvhNodeOffset,
            bvhNodes.length,
            0
        )
    }
    const objectsTypedArray = new Float32Array(sceneObjectSize * objectsArraySize)
    objectsTypedArray.set(objectsArray)

    const bvhNodeTypedArray = new Float32Array(bvhNodeSize * bvhNodeArraySize)
    bvhNodeTypedArray.set(bvhNodeArray)
    const bvhTriangleTypedArray = new Float32Array(meshArraySize)
    bvhTriangleTypedArray.set(bvhTriangleArray)

    const materialsArray: number[] = []
    for (const m of materials) {
        materialsArray.push(
            ...m.baseColor.toArray(),
            1,
            ...m.emissive,
            m.material.emissiveIntensity,
            m.metallic,
            m.roughness,
            m.ior,
            m.transmission
        )
    }
    const materialsTypedArray = new Float32Array(sceneMaterialSize * materialsArraySize)
    materialsTypedArray.set(materialsArray)

    const cameraArray = [
        ...camera.matrixWorld.toArray(),
        ...camera.rotation.toArray(),
        camera.sensorWidth,
        camera.focalLength,
        camera.focus,
        camera.fstop
    ]
    // TODO: optimize, too much copying
    const storageBufferArray = [
        ...indexArray,
        ...positionArray,
        ...normalArray,
        ...uvArray,
        ...objectsTypedArray,
        ...materialsTypedArray,
        ...bvhNodeTypedArray,
        ...bvhTriangleTypedArray,
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
        code: applyTemplate(renderWgsl, { commons, computeOutputTextureSize })
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

const applyTemplate = (code: string, variables: Record<string, string | number>): string => {
    let applied = code
    for (const [name, variable] of Object.entries(variables)) {
        applied = applied.replaceAll(`\${${name}}`, variable.toString())
    }
    return applied
}

const commons = applyTemplate(commonsWgsl, {})
const bvh = applyTemplate(bvhWgsl, {})

main()
