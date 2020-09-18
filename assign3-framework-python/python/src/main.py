#! /usr/bin/env python3

from environment import Environment
import heapq
from draw import *


def greedySearch(env):

    q = [(straight_line_distance(env.start, env.goal), env.start, None)]
    parent = {}
    visited = set()
    in_q = set()
    while len(q) > 0:
        _, v, p = heapq.heappop(q)
        if v not in parent:
            parent[v] = p
        else:
            continue

        if v == env.goal:
            return construct_path(parent, env.goal)

        visited.add(v)
        for nv in findNextVertex(v, env):
            new_distance = straight_line_distance(nv, env.goal)
            if nv in visited or nv in in_q:
                continue

            heapq.heappush(q, (new_distance, nv, v))
            in_q.add(nv)

    return []


def uniformCostSearch(env):
    q = [(straight_line_distance(env.start, env.goal), env.start, None)]
    parent = {}
    visited = set()
    distance_from_start = {env.start: 0}
    while len(q) > 0:
        distance, v, p = heapq.heappop(q)
        distance = distance_from_start[v]
        if v not in parent:
            parent[v] = p
        else:
            continue

        if v == env.goal:
            return construct_path(parent, env.goal)

        visited.add(v)
        for nv in findNextVertex(v, env):
            new_distance = distance + straight_line_distance(v, nv)

            if nv in visited:
                continue

            if nv not in distance_from_start:
                distance_from_start[nv] = new_distance
            else:
                if new_distance < distance_from_start[nv]:
                    distance_from_start[nv] = new_distance

            heapq.heappush(q, (new_distance, nv, v))

    return []


def astarSearch(env):

    q = [(straight_line_distance(env.start, env.goal), env.start, None)]
    parent = {}
    visited = set()
    distance_from_start = {env.start: 0}
    while len(q) > 0:
        distance, v, p = heapq.heappop(q)
        distance = distance_from_start[v]
        if v not in parent:
            parent[v] = p
        else:
            continue

        if v == env.goal:
            return construct_path(parent, env.goal)

        visited.add(v)
        for nv in findNextVertex(v, env):
            new_distance = distance + straight_line_distance(v, nv) + straight_line_distance(nv, env.goal)

            if nv in visited:
                continue

            if nv not in distance_from_start:
                distance_from_start[nv] = distance + straight_line_distance(v, nv)
            else:
                if distance + straight_line_distance(v, nv) < distance_from_start[nv]:
                    distance_from_start[nv] = distance + straight_line_distance(v, nv)

            heapq.heappush(q, (new_distance, nv, v))

    return []


if __name__ == '__main__':
    env = Environment('output/environment.txt')
    print("Loaded an environment with {} obstacles.".format(len(env.obstacles)))

    searches = {
        'greedy': greedySearch,
        'uniformcost': uniformCostSearch,
        'astar': astarSearch
    }

    for name, fun in searches.items():
        print("Attempting a search with " + name)
        Environment.printPath(name, fun(env))
