import { Box3, Triangle, Vector3 } from 'three'
import { SceneObject, bvhDepth, bvhSplitAccuracy } from '.'

export const axes = ['x', 'y', 'z'] as const
export const axisIndex = {
    x: 0,
    y: 1,
    z: 2
}
export type Axis = (typeof axes)[number]

export type BvhNode = {
    object: SceneObject
    box: Box3
} & (
    | {
          type: 'node'
          left: BvhNode
          right: BvhNode
      }
    | {
          type: 'leaf'
          triangles: number[]
      }
)

export type SplitResult = {
    splitAxis: Axis
    splitPoint: number
    left: number[]
    leftBox: Box3
    right: number[]
    rightBox: Box3
    cost: number
}

export const buildBvh = (object: SceneObject): BvhNode => {
    const triangleIdxs = []
    for (let fi = 0; fi < object.indexCount / 3; fi++) {
        triangleIdxs.push(fi)
    }
    const root: BvhNode = {
        object,
        box: makeBox(object, triangleIdxs),
        type: 'leaf',
        triangles: triangleIdxs
    }

    return splitBvh(root, 0)
}

export const optimalSplit = (node: BvhNode): SplitResult | undefined => {
    if (node.type === 'node') throw Error()
    let best: SplitResult | undefined = undefined

    for (const splitAxis of axes) {
        const axisLength = node.box.getSize(new Vector3())[splitAxis]
        const axisStart = node.box.min[splitAxis]
        // check more slices per vertex when BVH node's vertex count is low
        // https://www.desmos.com/calculator/5lwf9tbwym
        const accuracy = bvhSplitAccuracy / (node.object.indexCount + bvhSplitAccuracy)
        const cuts = Math.ceil(node.object.indexCount * accuracy)
        for (let cut = 0; cut < cuts; cut++) {
            const splitPoint = axisStart + axisLength * (cut / cuts)
            const left: number[] = []
            const right: number[] = []
            for (const ti of node.triangles) {
                const t = node.object.triangles[ti]
                if (t.a[splitAxis] >= splitPoint || t.b[splitAxis] >= splitPoint || t.c[splitAxis] >= splitPoint) {
                    left.push(ti)
                } else {
                    right.push(ti)
                }
            }
            if (left.length === 0 || right.length === 0) continue

            const leftBox = makeBox(node.object, left)
            const rightBox = makeBox(node.object, right)
            const cost = bvhCost(leftBox, left.length) + bvhCost(rightBox, right.length)
            if (best === undefined || cost < best.cost) {
                best = {
                    splitAxis,
                    splitPoint,
                    left: left,
                    leftBox,
                    right: right,
                    rightBox,
                    cost
                }
            }
        }
    }

    return best
}

export const splitBvh = (node: BvhNode, depth: number): BvhNode => {
    if (depth >= bvhDepth) {
        console.warn('hit the bvh depth limit', node, depth)
        return node
    }
    if (node.type === 'node') throw Error()
    const object = node.object

    const nodeNew: BvhNode = {
        type: 'node',
        object,
        box: node.box,
        left: undefined as any,
        right: undefined as any
    }

    const split = optimalSplit(node)
    if (!split) return node

    if (split.left.length > 0) {
        const left: BvhNode = {
            object,
            box: split.leftBox,
            type: 'leaf',
            triangles: split.left
        }
        nodeNew.left = splitBvh(left, depth + 1)
    }
    if (split.right.length > 0) {
        const right: BvhNode = {
            object,
            box: split.rightBox,
            type: 'leaf',
            triangles: split.right
        }
        nodeNew.right = splitBvh(right, depth + 1)
    }
    if (!nodeNew.left || node.triangles.length === split.right.length) return nodeNew.right!
    if (!nodeNew.right || node.triangles.length === split.left.length) return nodeNew.left!

    return nodeNew
}

export const bvhCost = (box: Box3, triangles: number) => {
    const size = box.getSize(new Vector3())
    return size.x * size.y * size.z * triangles
}

export const makeBox = (object: SceneObject, triangles: number[]): Box3 => {
    if (triangles.length === 0) return new Box3()
    const t0 = object.triangles[triangles[0]]
    const min = t0.a.clone()
    const max = t0.a.clone()
    for (const t of triangles) {
        const triangle = object.triangles[t]
        min.x = Math.min(min.x, triangle.a.x, triangle.b.x, triangle.c.x)
        min.y = Math.min(min.y, triangle.a.y, triangle.b.y, triangle.c.y)
        min.z = Math.min(min.z, triangle.a.z, triangle.b.z, triangle.c.z)
        max.x = Math.max(max.x, triangle.a.x, triangle.b.x, triangle.c.x)
        max.y = Math.max(max.y, triangle.a.y, triangle.b.y, triangle.c.y)
        max.z = Math.max(max.z, triangle.a.z, triangle.b.z, triangle.c.z)
    }
    return new Box3(min, max)
}

export const triangleByIndex = (object: SceneObject, index: number): Triangle => {
    const triangle = [new Vector3(), new Vector3(), new Vector3()]
    for (let v = 0; v < 3; v++) {
        const triIndex = object.index[3 * index + v]
        triangle[v] = new Vector3(
            object.position[3 * triIndex],
            object.position[3 * triIndex + 1],
            object.position[3 * triIndex + 2]
        )
    }
    return new Triangle(...triangle)
}

export const traverseBfs = (root: BvhNode): BvhNode[] => {
    const result: BvhNode[] = []
    const queue = [root]
    while (queue.length > 0) {
        const node = queue.splice(0, 1)[0]
        result.push(node)
        if (node.type === 'node') {
            queue.push(node.left, node.right)
        }
    }
    return result
}
