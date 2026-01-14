import './index.css'

const workgroupSize = [8, 8]
const resolution = [100, 100]
const format: GPUTextureFormat = 'rgba8unorm'

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let computePipeline: GPUComputePipeline
let computeOutputTexture: GPUTexture
let computeBindGroup: GPUBindGroup
let renderPipeline: GPURenderPipeline
let renderBindGroup: GPUBindGroup
let clipVertexBuffer: GPUBuffer

const wgsl = String.raw

const main = async (): Promise<void> => {
    if (!navigator.gpu) {
        alert('WebGPU is not supported')
        return
    }

    device = (await initDevice())!
    if (!device) {
        alert('no WebGPU device')
        return
    }
    console.debug(device)

    canvas = document.getElementById('canvas') as HTMLCanvasElement
    ctx = canvas.getContext('webgpu')!
    console.debug(ctx)
    ctx.configure({ device, format, alphaMode: 'premultiplied' })

    const resize = () => {
        const dpr = window.devicePixelRatio
        canvas.width = window.innerWidth * dpr
        canvas.height = window.innerHeight * dpr
    }
    window.addEventListener('resize', resize)
    resize()

    await initCompute()
    await initRender()

    // requestAnimationFrame(loop)
    await update()
}

const initDevice = async (): Promise<GPUDevice | undefined> => {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return undefined
    return await adapter.requestDevice()
}

const initCompute = async () => {
    const computeModule = device.createShaderModule({
        code: wgsl`
const resolution = vec2u(100, 100);

@group(0) @binding(0) var out: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(${workgroupSize.join(',')})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= resolution.x || gid.y >= resolution.y) { return; }
  let u = f32(gid.x) / f32(resolution.x - 1);
  let v = f32(gid.y) / f32(resolution.y - 1);
  textureStore(out, vec2<u32>(gid.x, gid.y), vec4<f32>(u, v, 0.5, 1.0));
}
`
    })

    computePipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: computeModule, entryPoint: 'main' }
    })

    computeOutputTexture = device.createTexture({
        size: [resolution[0], resolution[1], 1],
        format,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    })

    computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
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
// @group(0) @binding(0) var computeTexture: texture_2d<f32>;
// @group(0) @binding(1) var computeSampler: sampler;

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
};

@vertex
fn mainVertex(@location(0) position: vec2f) -> VertexOut {
    return VertexOut(vec4f(position, 0., 1.), position);
}

@fragment
fn mainFragment(vout: VertexOut) -> @location(0) vec4f {
    let uv = vout.uv.xy * .5 + .5;
    return vec4f(uv.xy, 0., 1.);
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
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list' }
    })

    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            // { binding: 0, resource: computeOutputTexture.createView() },
            // { binding: 1, resource: sampler }
        ]
    })
}

const loop = async () => {
    await update()
    requestAnimationFrame(update)
}

const update = async () => {
    const start = performance.now()

    compute()
    draw()
    await device.queue.onSubmittedWorkDone()

    const end = performance.now()
    document.getElementById('delta')!.innerText = (end - start).toFixed(2).padStart(5, ' ')
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
