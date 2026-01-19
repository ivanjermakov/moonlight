import { Box3, Triangle, Vector3 } from 'three'
import { SceneObject, bvhDepth } from '.'

export const axes = ['x', 'y', 'z'] as const
export type Axis = (typeof axes)[number]

export type BvhNode = {
    object: SceneObject
    box: Box3
} & (
    | {
          type: 'node'
          left: BvhNode | undefined
          right: BvhNode | undefined
      }
    | {
          type: 'leaf'
          triangleIdxs: number[]
          triangles: Triangle[]
      }
)

export const buildBvh = (object: SceneObject): BvhNode => {
    const triangleIdxs = []
    for (let fi = 0; fi < object.indexCount / 3; fi++) {
        triangleIdxs.push(fi)
    }
    const triangles = triangleIdxs.map(i => triangleByIndex(object, i))
    const root: BvhNode = {
        object,
        box: makeBox(triangles),
        type: 'leaf',
        triangleIdxs,
        triangles
    }

    return splitBvh(root, 0)
}

export type SplitResult = {
    splitAxis: Axis
    splitPoint: number
    left: Triangle[]
    leftIdxs: number[]
    leftBox: Box3
    right: Triangle[]
    rightIdxs: number[]
    rightBox: Box3
    cost: number
}

export const optimalSplit = (node: BvhNode): SplitResult | undefined => {
    if (node.type === 'node') throw Error()
    let best: SplitResult | undefined = undefined

    for (const splitAxis of axes) {
        let axisIndex: number
        // biome-ignore format:
        switch (splitAxis) {
                case 'x': axisIndex = 0; break;
                case 'y': axisIndex = 1; break;
                case 'z': axisIndex = 2; break;
            }
        const splitAxisPoints = new Set<number>()
        for (let vi = 0; vi < node.object.positionCount / 3; vi++) {
            splitAxisPoints.add(node.object.position[3 * vi + axisIndex])
        }
        for (const splitPoint of splitAxisPoints) {
            const left: Triangle[] = []
            const leftIdxs: number[] = []
            const right: Triangle[] = []
            const rightIdxs: number[] = []
            for (let i = 0; i < node.triangles.length; i++) {
                const ti = node.triangleIdxs[i]
                const t = node.triangles[i]
                if (t.a[splitAxis] >= splitPoint || t.c[splitAxis] >= splitPoint || t.c[splitAxis] >= splitPoint) {
                    left.push(t)
                    leftIdxs.push(ti)
                } else {
                    right.push(t)
                    rightIdxs.push(ti)
                }
            }
            if (left.length === 0 || right.length === 0) continue
            if (left.length === node.triangles.length || right.length === node.triangles.length) continue

            const leftBox = makeBox(left)
            const rightBox = makeBox(right)
            const cost = bvhCost(leftBox, left.length) + bvhCost(rightBox, right.length)
            if (best === undefined || cost < best.cost) {
                best = {
                    splitAxis,
                    splitPoint,
                    left,
                    leftIdxs,
                    leftBox,
                    right,
                    rightIdxs,
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
        left: undefined,
        right: undefined
    }

    const split = optimalSplit(node)
    if (!split) return node

    if (split.left.length > 0) {
        const left: BvhNode = {
            object,
            box: split.leftBox,
            type: 'leaf',
            triangleIdxs: split.leftIdxs,
            triangles: split.left
        }
        nodeNew.left = splitBvh(left, depth + 1)
    }
    if (split.right.length > 0) {
        const right: BvhNode = {
            object,
            box: split.rightBox,
            type: 'leaf',
            triangleIdxs: split.rightIdxs,
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

export const makeBox = (triangles: Triangle[]): Box3 => {
    if (triangles.length === 0) return new Box3()
    const min = triangles[0].a.clone()
    const max = triangles[0].a.clone()
    for (const triangle of triangles) {
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
