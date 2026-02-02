class SliceViewer:
    def __init__(self):
        self.image = None

    def set_image(self, image):
        self.image = image


class PointSelector:
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point)