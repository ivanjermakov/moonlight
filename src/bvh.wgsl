// aabbMin and aabbMax are split, otherwise nested struct makes struct 4 floats bigger due to alignment
struct BvhNode {
    aabbMin: vec3f,
    // start index into array of leaf items if type leaf, node index if type node
    indexOrLeafOffset: f32,
    aabbMax: vec3f,
    count: f32,
}

const bvhDepth = ${bvhDepth};

fn castRay(ray: Ray) -> RayCast {
    var rayCast = RayCast();
    rayCast.distance = maxDistance;

    var stack: array<u32, bvhDepth + 1>;
    var stackIdx = 0u;

    for (var i = 0u; i < u32(store.objectCount); i++) {
        let object = store.objects[i];

        if (intersectAabb(ray, object.boundingBox) >= maxDistance) {
            continue;
        }

        let rootIdx = u32(object.bvhOffset);
        stack[stackIdx] = rootIdx;
        stackIdx++;

        while stackIdx > 0 {
            stackIdx--;
            let bvhNodeIdx = stack[stackIdx];
            let bvhNode = store.bvhNode[bvhNodeIdx];
            if bvhNode.count > 0 {
                let indexOffset = u32(object.indexOffset);
                let vertexOffset = u32(object.vertexOffset);
                let triangleOffset = u32(bvhNode.indexOrLeafOffset);
                for (var bvhTriangleIdx = 0u; bvhTriangleIdx < u32(bvhNode.count); bvhTriangleIdx++) {
                    let triangleIdx = u32(store.bvhTriangle[triangleOffset + bvhTriangleIdx]);
                    var triangle: array<vec3f, 3>;
                    for (var v = 0u; v < 3; v++) {
                        let vertIdx = u32(store.index[indexOffset + 3 * triangleIdx + v]);
                        triangle[v] = vec3f(
                            store.position[3 * (vertexOffset + vertIdx)],
                            store.position[3 * (vertexOffset + vertIdx) + 1],
                            store.position[3 * (vertexOffset + vertIdx) + 2],
                        );
                    }
                    let intersection = intersectTriangle(ray, triangle);
                    if intersection.hit {
                        let d = distance(intersection.point, ray.origin);
                        if d < rayCast.distance {
                            var triNormals: array<vec3f, 3>;
                            for (var v = 0u; v < 3; v++) {
                                let vertIdx = u32(store.index[indexOffset + 3 * triangleIdx + v]);
                                let vertIdxGlobal = 3 * (vertexOffset + vertIdx);
                                triNormals[v] = vec3f(
                                    store.normal[vertIdxGlobal],
                                    store.normal[vertIdxGlobal + 1],
                                    store.normal[vertIdxGlobal + 2],
                                );
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
                            rayCast.face = triangleIdx;
                            rayCast.distance = d;
                        }
                    }
                }
            } else {
                let idxLeft = u32(bvhNode.indexOrLeafOffset);
                let left = store.bvhNode[idxLeft];
                let aabbLeft = Aabb(left.aabbMin, left.aabbMax);
                let distLeft = intersectAabb(ray, aabbLeft);
                let idxRight = u32(bvhNode.indexOrLeafOffset + 1);
                let right = store.bvhNode[idxRight];
                let aabbRight = Aabb(right.aabbMin, right.aabbMax);
                let distRight = intersectAabb(ray, aabbRight);

                let leftCloser = distLeft < distRight;

                let distNear = select(distLeft, distRight, !leftCloser); 
                let idxNear = select(idxLeft, idxRight, !leftCloser);
                let distFar = select(distLeft, distRight, leftCloser); 
                let idxFar = select(idxLeft, idxRight, leftCloser);

                if distFar < rayCast.distance {
                    stack[stackIdx] = idxFar;
                    stackIdx++;
                }
                if distNear < rayCast.distance {
                    stack[stackIdx] = idxNear;
                    stackIdx++;
                }
            }
        }
    }
    return rayCast;
}

