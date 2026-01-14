import './index.css'

const workgroupSize = [8, 8]
const resolution = [32, 32]
const formatComputeTexture: GPUTextureFormat = 'rgba16float'

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let computePipeline: GPUComputePipeline
let computeOutputTexture: GPUTexture
let computeBindGroup: GPUBindGroup
let renderPipeline: GPURenderPipeline
let renderBindGroup: GPUBindGroup
let clipVertexBuffer: GPUBuffer
let formatCanvas: GPUTextureFormat

let frameStart: number = 0

const wgsl = String.raw

const main = async (): Promise<void> => {
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
        console.debug('resize', canvas.width, canvas.height)
    }
    window.addEventListener('resize', resize)
    resize()

    await initCompute()
    await initRender()

    requestAnimationFrame(loop)
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
const resolution = vec2u(${resolution.join(',')});

@group(0) @binding(0) var out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${workgroupSize.join(',')})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= resolution.x || gid.y >= resolution.y) { return; }
  let uv = vec3f(gid).xy / vec2f(resolution);
  textureStore(out, vec2u(gid.xy), vec4<f32>(uv, 0., 1.));
}
`
    })

    // needed because rgba16float is not the default choice for storage textures
    const layout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: formatComputeTexture } }]
    })

    computePipeline = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: computeModule, entryPoint: 'main' }
    })

    computeOutputTexture = device.createTexture({
        size: [resolution[0], resolution[1], 1],
        format: formatComputeTexture,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    })

    computeBindGroup = device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: computeOutputTexture.createView() }]
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

    device.queue.writeBuffer(clipVertexBuffer, 0, clipPlane, 0, clipPlane.length)

    const renderModule = device.createShaderModule({
        code: wgsl`
const resolution = vec2u(${resolution.join(',')});

@group(0) @binding(0) var computeTexture: texture_2d<f32>;
@group(0) @binding(1) var computeSampler: sampler;

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
    return textureSample(computeTexture, computeSampler, vout.uv);
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
        minFilter: 'nearest',
        addressModeU: 'repeat',
        addressModeV: 'repeat'
    })
    renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: computeOutputTexture.createView() },
            { binding: 1, resource: sampler }
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

const compute = () => {
    const commandEncoder = device.createCommandEncoder()
    const pass = commandEncoder.beginComputePass()
    pass.setPipeline(computePipeline)
    pass.setBindGroup(0, computeBindGroup)
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
    passEncoder.draw(6)
    passEncoder.end()
    device.queue.submit([commandEncoder.finish()])
}

main()
