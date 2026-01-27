import { Matrix4, Vector4 } from 'three'

export const transformPointArray = (a: Float32Array, mat: Matrix4): Float32Array => {
    const result = new Float32Array(a.length)
    for (let i = 0; i < a.length; i += 3) {
        const res = new Vector4(a[i], a[i + 1], a[i + 2], 1).applyMatrix4(mat)
        result[i] = res.x
        result[i + 1] = res.y
        result[i + 2] = res.z
    }
    return result
}

export const transformDirArray = (a: Float32Array, mat: Matrix4): Float32Array => {
    const result = new Float32Array(a.length)
    for (let i = 0; i < a.length; i += 3) {
        const res = new Vector4(a[i], a[i + 1], a[i + 2], 0).applyMatrix4(mat).normalize().toArray()
        result[i] = res[0]
        result[i + 1] = res[1]
        result[i + 2] = res[2]
    }
    return result
}

export const transformDir4Array = (a: Float32Array, mat: Matrix4): Float32Array => {
    const result = new Float32Array(a.length)
    for (let i = 0; i < a.length; i += 4) {
        const res = new Vector4(a[i], a[i + 1], a[i + 2], a[i + 3]).applyMatrix4(mat).normalize().toArray()
        result[i] = res[0]
        result[i + 1] = res[1]
        result[i + 2] = res[2]
        result[i + 3] = res[3]
    }
    return result
}
