import './index.css'

let device: GPUDevice
let canvas: HTMLCanvasElement
let ctx: GPUCanvasContext
let renderPipeline: GPURenderPipeline
let vertexBuffer: GPUBuffer

const wgsl = String.raw

const shaders = wgsl`
struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f
}

@vertex
fn mainVertex(@location(0) position: vec4f, @location(1) color: vec4f) -> VertexOut {
    var output : VertexOut;
    output.position = position;
    output.color = color;
    return output;
}

@fragment
fn mainFragment(fragData: VertexOut) -> @location(0) vec4f {
    return fragData.color;
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
    // x, y, z, w, r, g, b, a
    const vertices = new Float32Array([
         0.0,  0.6, 0, 1,   1, 0, 0, 1,
        -0.5, -0.6, 0, 1,   0, 1, 0, 1,
         0.5, -0.6, 0, 1,   0, 0, 1, 1
    ])

    vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    })

    device.queue.writeBuffer(vertexBuffer, 0, vertices, 0, vertices.length)

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
                            format: 'float32x4'
                        },
                        {
                            // color
                            shaderLocation: 1,
                            offset: 16,
                            format: 'float32x4'
                        }
                    ],
                    arrayStride: 32,
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
    passEncoder.draw(3)
    passEncoder.end()
    device.queue.submit([commandEncoder.finish()])
}

main()
