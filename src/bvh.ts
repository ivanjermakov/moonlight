import { Box3, Triangle, Vector3 } from 'three'
import { SceneObject, bvhDepth } from '.'

export const axes = ['x', 'y', 'z'] as const
export const axisIndex = {
    x: 0,
    y: 1,
    z: 2
}
export type Axis = (typeof axes)[number]

export type BvhNode = {
    box: Box3
} & (
    | {
          type: 'node'
          left: BvhNode
          right: BvhNode
      }
    | {
          type: 'leaf'
          index: number[]
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

export const buildBvhTris = (object: SceneObject, splitAccuracy: number): BvhNode => {
    const boxes = object.triangles.map(t => new Box3().setFromPoints([t.a, t.b, t.c]))
    const indexer = (i: number) => boxes[i]
    const triangleIdxs = Array(object.indexCount / 3)
        .fill(0)
        .map((_, i) => i)
    const root: BvhNode = {
        box: makeBox(indexer, triangleIdxs),
        type: 'leaf',
        index: triangleIdxs
    }
    return splitBvh(root, indexer, splitAccuracy, 0)
}

export const buildBvhObjects = (objects: SceneObject[], splitAccuracy: number): BvhNode => {
    const indexer = (i: number) => objects[i].boundingBox
    const objectIdxs = Array(objects.length)
        .fill(0)
        .map((_, i) => i)
    const root: BvhNode = {
        box: makeBox(indexer, objectIdxs),
        type: 'leaf',
        index: objectIdxs
    }
    return splitBvh(root, indexer, splitAccuracy, 0)
}

/**
 * For further optimization, consider binning https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
 */
export const optimalSplit = (
    node: BvhNode,
    indexer: (i: number) => Box3,
    splitAccuracy: number
): SplitResult | undefined => {
    if (node.type === 'node') throw Error()
    let best: SplitResult | undefined = undefined
    const vertCount = node.index.length

    for (const splitAxis of axes) {
        const axisLength = node.box.getSize(new Vector3())[splitAxis]
        const axisStart = node.box.min[splitAxis]
        // check less slices per vertex when BVH node's vertex count is high
        // https://www.desmos.com/calculator/5lwf9tbwym
        const accuracy = splitAccuracy / (vertCount + splitAccuracy)
        const cuts = Math.ceil(vertCount * accuracy)
        for (let cut = 0; cut < cuts; cut++) {
            const splitPoint = axisStart + axisLength * (cut / cuts)
            const left: number[] = []
            const right: number[] = []
            for (const ti of node.index) {
                if (indexer(ti).min[splitAxis] < splitPoint) {
                    left.push(ti)
                } else {
                    right.push(ti)
                }
            }
            if (left.length === 0 || right.length === 0) continue

            const leftBox = makeBox(indexer, left)
            const rightBox = makeBox(indexer, right)
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

export const splitBvh = (
    node: BvhNode,
    indexer: (i: number) => Box3,
    splitAccuracy: number,
    depth: number
): BvhNode => {
    if (depth >= bvhDepth) {
        console.warn('hit the bvh depth limit', node, depth)
        return node
    }
    if (node.type === 'node') throw Error()

    const nodeNew: BvhNode = {
        type: 'node',
        box: node.box,
        left: undefined as any,
        right: undefined as any
    }

    const split = optimalSplit(node, indexer, splitAccuracy)
    if (!split) return node

    if (split.left.length > 0) {
        const left: BvhNode = {
            box: split.leftBox,
            type: 'leaf',
            index: split.left
        }
        nodeNew.left = splitBvh(left, indexer, splitAccuracy, depth + 1)
    }
    if (split.right.length > 0) {
        const right: BvhNode = {
            box: split.rightBox,
            type: 'leaf',
            index: split.right
        }
        nodeNew.right = splitBvh(right, indexer, splitAccuracy, depth + 1)
    }
    if (!nodeNew.left || node.index.length === split.right.length) return nodeNew.right!
    if (!nodeNew.right || node.index.length === split.left.length) return nodeNew.left!

    return nodeNew
}

export const bvhCost = (box: Box3, triangles: number) => {
    const size = box.getSize(new Vector3())
    const halfSurfaceArea = size.x * size.y + size.x * size.z + size.y * size.z
    return halfSurfaceArea * triangles
}

export const makeBox = (indexer: (i: number) => Box3, index: number[]): Box3 => {
    if (index.length === 0) return new Box3()
    const t0 = indexer(index[0])
    const box = t0.clone()
    for (const t of index) {
        box.union(indexer(t))
    }
    return box
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

export const longestAxis = (v: Vector3): Axis => {
    const longestSize = Math.max(v.x, v.y, v.z)
    if (v.x === longestSize) return 'x'
    if (v.y === longestSize) return 'y'
    return 'z'
}
