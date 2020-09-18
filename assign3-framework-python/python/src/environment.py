from polygon import Polygon
from vector2d import Vector2D


class Environment:
    def __init__(self, filename):
        self.width = 800
        self.height = 600
        self.start = Vector2D(0, 0)
        self.goal = Vector2D(800, 600)
        self.obstacles = []

        f = open(filename, 'r')
        envtxt = f.readlines()
        f.close()
        polygonstxt, *resttxt = envtxt
        polygons = int(polygonstxt)
        for polygon_number in range(polygons):
            ntxt, *resttxt = resttxt
            n = int(ntxt)
            p = Polygon(n)
            for line in resttxt[:n]:
                [x, y] = [float(x) for x in line.split()]
                p.vertices.append(Vector2D(x, y))
            resttxt = resttxt[n:]
            self.obstacles.append(p)

    @staticmethod
    def printPath(searchName, path):
        with open('output/' + searchName + '.js', 'w') as f:
            f.write('window.' + searchName + ' =\n\t[\n')
            for v in path:
                f.write('\t\t[{}, {}],\n'.format(v.x, v.y))
            f.write('\t];\n')
