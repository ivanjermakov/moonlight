struct BvhNode {
    aabb: Aabb,
    // into store.bvhTriangle
    triangleOffset: f32,
    triangleCount: f32,
    // into store.bvhNode
    // store.bvhNode[leafOffset] is node.left
    // store.bvhNode[leafOffset + 1] is node.right
    // only valid if triangleCount == 0
    leafOffset: f32,
    p1: f32,
}

fn castRayBvh(ray: Ray) -> RayCast {
    var rayCast = RayCast();
    rayCast.distance = maxDistance;
    for (var i = 0u; i < u32(store.objectCount); i++) {
        let object = store.objects[i];
        let root = store.bvhNode[u32(object.bvhOffset)];
        if !intersectAabb(ray, root.aabb) {
            continue;
        }
        let indexOffset = u32(object.indexOffset);
        let vertexOffset = u32(object.vertexOffset);
        for (var fi = 0u; fi < u32(object.indexCount / 3); fi++) {
            var triangle: array<vec3f, 3>;
            for (var v = 0u; v < 3; v++) {
                let triIndex = u32(store.index[indexOffset + 3 * fi + v]);
                let triIndexGlobal = 3 * (vertexOffset + triIndex);
                let trianglePos = vec3f(
                    store.position[triIndexGlobal],
                    store.position[triIndexGlobal + 1],
                    store.position[triIndexGlobal + 2],
                );
                triangle[v] = trianglePos;
            }
            let intersection = intersectTriangle(ray, triangle);
            if intersection.hit {
                let d = distance(intersection.point, ray.origin);
                if d < rayCast.distance {
                    var triNormals: array<vec3f, 3>;
                    for (var v = 0u; v < 3; v++) {
                        let triIndex = u32(store.index[indexOffset + 3 * fi + v]);
                        let triIndexGlobal = 3 * (vertexOffset + triIndex);
                        let vertexNormal = vec3f(
                            store.normal[triIndexGlobal],
                            store.normal[triIndexGlobal + 1],
                            store.normal[triIndexGlobal + 2],
                        );
                        triNormals[v] = vertexNormal;
                    }
                    let u = intersection.uv.x;
                    let v = intersection.uv.y;
                    let normal = normalize(
                        triNormals[0] + (triNormals[1] - triNormals[0]) * u + (triNormals[2] - triNormals[0]) * v
                    );

                    if dot(normal, ray.dir) > 0 {
                        let material = store.materials[u32(object.material)];
                        if material.transmission == 0 {
                            continue;
                        }
                    }

                    rayCast.intersection = intersection;
                    rayCast.normal = normal;
                    rayCast.object = i;
                    rayCast.face = fi;
                    rayCast.distance = d;
                }
            }
        }
    }
    return rayCast;
}

