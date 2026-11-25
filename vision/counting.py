"""Dependency-free line-crossing logic with deadband."""


class LineCounter:
    def __init__(self, line_y=0.55, margin=0.015):
        if not 0 < margin < min(line_y, 1 - line_y):
            raise ValueError("Boundary and margin must fit inside normalized frame")
        self.line_y = line_y
        self.margin = margin
        self.side = {}
        self.counted = set()

    def observe(self, key, bottom_y):
        """Return 'up' or 'down' once when a tracked object crosses the line.
        Ignores the deadband and never counts objects first observed inside it.
        """
        if key in self.counted:
            return None

        if bottom_y < self.line_y - self.margin:
            current = -1
        elif bottom_y > self.line_y + self.margin:
            current = 1
        else:
            return None

        previous = self.side.get(key)
        self.side[key] = current

        if previous is not None and previous != current:
            self.counted.add(key)
            self.side.pop(key, None)
            return "down" if current == 1 else "up"

        return None
