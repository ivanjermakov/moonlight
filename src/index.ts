import './index.css'

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let renderPipeline: GPURenderPipeline
let vertexBuffer: GPUBuffer

const wgsl = String.raw

const shaders = wgsl`
struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>
};

@vertex
fn mainVertex(@location(0) position: vec2f) -> VertexOut {
    var out: VertexOut;
    out.pos = vec4f(position, 0., 1.);
    out.uv = position;
    return out;
}

@fragment
fn mainFragment(vout: VertexOut) -> @location(0) vec4f {
    let uv = vout.uv.xy * .5 + .5;
    return vec4f(uv.xy, 0., 1.);
}
`

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
    ctx.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'premultiplied' })

    const resize = () => {
        const dpr = window.devicePixelRatio
        canvas.width = window.innerWidth * dpr
        canvas.height = window.innerHeight * dpr
    }
    window.addEventListener('resize', resize)
    resize()

    const shaderModule = device.createShaderModule({ code: shaders })

    // biome-ignore format:
    const clipPlane = new Float32Array([
         -1, -1,
          1,  1,
         -1,  1,
         -1, -1,
          1, -1,
          1,  1,
    ])

    vertexBuffer = device.createBuffer({
        size: clipPlane.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    })

    device.queue.writeBuffer(vertexBuffer, 0, clipPlane, 0, clipPlane.length)

    renderPipeline = await device.createRenderPipelineAsync({
        vertex: {
            module: shaderModule,
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
            module: shaderModule,
            entryPoint: 'mainFragment',
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
        },
        primitive: {
            topology: 'triangle-list'
        },
        layout: 'auto'
    })

    requestAnimationFrame(update)
}

const initDevice = async (): Promise<GPUDevice | undefined> => {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return undefined
    return await adapter.requestDevice()
}

const update = () => {
    draw()
    requestAnimationFrame(update)
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
    passEncoder.setVertexBuffer(0, vertexBuffer)
    passEncoder.draw(6)
    passEncoder.end()
    device.queue.submit([commandEncoder.finish()])
}

main()
