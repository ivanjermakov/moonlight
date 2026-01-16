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
import commonsWgsl from './commons.wgsl?raw'
import computeWgsl from './compute.wgsl?raw'
import './index.css'
import renderWgsl from './render.wgsl?raw'

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
type SceneName = 'cornell-box' | 'rough-metallic'
const sceneName = 'rough-metallic' as SceneName

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

const main = async (): Promise<void> => {
    const gltfPath = `/${sceneName}.glb`
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
        a.download = `${sceneName}-${new Date()
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
        code: applyTemplate(computeWgsl, { commons, maxBounces, workgroupSize: workgroupSize.join(',') })
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
            // TODO: explain
            Math.sqrt(m.roughness),
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
        code: applyTemplate(renderWgsl, { commons, computeOutputTextureSize })
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

const applyTemplate = (code: string, variables: Record<string, string | number>): string => {
    let applied = code
    for (const [name, variable] of Object.entries(variables)) {
        applied = applied.replaceAll(`\${${name}}`, variable.toString())
    }
    return applied
}

const commons = applyTemplate(commonsWgsl, {
    meshArraySize,
    objectsArraySize,
    materialsArraySize
})

main()
