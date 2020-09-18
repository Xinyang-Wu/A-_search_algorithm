
def findVertexAndSegment(env):
    segments = []
    vertices = []

    for obstacle in env.obstacles:
        for v in obstacle.vertices:
            vertices.append(v)

        for i in range(len(obstacle.vertices)):
            v1, v2 = obstacle.vertices[i], obstacle.vertices[( i +1) % len(obstacle.vertices)]
            segments.append((v1, v2))
    vertices.append(env.start)
    vertices.append(env.goal)
    return vertices, segments


def onSegment(p, q, r):
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):

    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def cross(seg1, seg2):
    v_set = set()
    p1, q1 = seg1
    p2, q2 = seg2

    v_set.add(p1)
    v_set.add(p2)
    v_set.add(q1)
    v_set.add(q2)

    if len(v_set) == 3:
        return False

    # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0) and onSegment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0) and onSegment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0) and onSegment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0) and onSegment(p2, q1, q2):
        return True

    # If none of the cases
    return False


def findNextVertex(v, env):
    ob = None
    next_v = [env.goal]
    for obstacle in env.obstacles:
        if v in obstacle.vertices:
            ob = obstacle
        else:
            next_v.extend(obstacle.vertices)

    all_v, all_seg = findVertexAndSegment(env)
    legal_vertices = []
    for nv in next_v:
        c = False
        for seg in all_seg:     # check every segment
            if cross((v, nv), seg):
                c = True
        if not c:
            legal_vertices.append(nv)

    if v != env.start and v != env.goal:
        i = ob.vertices.index(v)
        legal_vertices.append(ob.vertices[i - 1])
        legal_vertices.append(ob.vertices[(i + 1) % len(ob.vertices)])

    return legal_vertices


def straight_line_distance(v1, v2):
    return v1.subtract(v2).abs()


def construct_path(parent, goal):
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]
