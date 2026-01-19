struct BvhNode {
    aabb: Aabb,
    // into store.bvhTriangle
    triangleOffset: f32,
    triangleCount: f32,
    // into store.bvhNode
    // only valid if triangleCount == 0
    leafOffset: f32,
    p1: f32,
}
