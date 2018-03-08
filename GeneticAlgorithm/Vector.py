# python import based off of PVector library
# source: http://natureofcode.com/

import math


class Vector:
    """Data type for vectors in 2-3 dimensions"""
    def __init__(self, x_=0, y_=0, z_=0):
        """Arguments: x value, y value, z value (default = 0)"""
        self.x = x_
        self.y = y_
        self.z = z_

    def __str__(self):
        """Returns non-zero fields in a string"""
        if self.z == 0:
            return '%f,%f' % (self.x, self.y)
        elif self.x == 0:
            return '%f,%f' % (self.y, self.z)
        elif self.y == 0:
            return '%f,%f' % (self.x, self.z)

    def int(self):
        """Makes all attributes integers"""
        self.x = int(self.x)
        self.y = int(self.y)
        self.z = int(self.z)

    # method: sets vector values
    def set(self, x_=0, y_=0, z_=0, other=None):
        """Sets vector attributes
            Keyword Arguments:
            Either:
            x,y,z (default = 0)
            Or:
            other -- a Vector (default = None)
            """
        if other is None:
            self.x = x_
            self.y = y_
            self.z = z_
        elif x_ is 0 and y_ is 0 and z_ is 0:
            self.x = other.x
            self.y = other.y
            self.z = other.z
        else:
            print("WARNING: You passed both component arguments and a Vector argument\nSetting values to the Vector")
            self.x = other.x
            self.y = other.y
            self.z = other.z

    # sets vector based on angle
    # ex: from_angle(angle, vector)
    @staticmethod
    def from_angle(angle, target=None):
        """Returns a new Vector with a specific heading using a radian angle argument"""
        if target is None:
            target = Vector(math.cos(angle), math.sin(angle))
        else:
            target.set = Vector(math.cos(angle), math.sin(angle))
        return target

    # static: copies vector to new vector
    def copy(self):
        """Returns a copy of the Vector"""
        return Vector(self.x, self.y, self.z)

    # static : magnitude of vector
    # ex: magnitude = self.mag()
    def mag(self):
        """Returns the magnitude of the Vector as a float"""
        mag = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        return mag

    # static : magnitude of vector squared
    # ex: magnitude = self.magSq()
    def mag_sq(self):
        """Returns the magnitude of the Vector squared as a float"""
        mag_squared = self.mag() ** 2
        return mag_squared

    # method: vector addition
    # ex: self.add(vector)
    def add(self, x_=0,y_=0,z_=0, other=None):
        """Adds each component to the Vector
            Keyword Arguments:
            Either:
            x,y,z (default = 0)
            Or:
            other -- a Vector (default = None)
        """
        if other is None:
            self.x += x_
            self.y += x_
            self.z += x_
        elif x_ is 0 and y_ is 0 and z_ is 0:
            self.x += other.x
            self.y += other.y
            self.z += other.z
        else:
            print("WARNING: You passed both component arguments and a Vector argument\nSetting values to the Vector")
            self.x += other.x
            self.y += other.y
            self.z += other.z

    # static: vector addition
    # ex: vector = self + vect
    def __add__(self, other):
        """Returns a new vector which is the result of adding the two vectors
            Arguments:
            Other -- a Vector
        """
        vector_ = Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        return vector_

    # method: vector subtraction
    # ex: self.sub(vector)
    def sub(self, x_=0, y_=0, z_=0, other=None):
        """Subtracts each component to the Vector
                    Keyword Arguments:
                    Either:
                    x,y,z (default = 0)
                    Or:
                    other -- a Vector (default = None)
                """
        if other is None:
            self.x -= x_
            self.y -= x_
            self.z -= x_
        elif x_ is 0 and y_ is 0 and z_ is 0:
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
        else:
            print("WARNING: You passed both component arguments and a Vector argument\nSetting values to the Vector")
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z

    # static: vector subtraction
    # ex: vector = self - vect
    def __sub__(self, other):
        """Subtracts the two vectors
            Arguments:
                Other -- The vector to be subtracted from the callee
        """
        vector_ = Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        return vector_

    # method : scalar multiplication
    # ex: self.mult(n)
    def mult(self, n):
        """Multiplies the vector by a scalar number
            Arguments:
                n -- the scalar number to multiply by
        """
        self.x = self.x * n
        self.y = self.y * n
        self.z = self.z * n

    # static : scalar multiplication
    # ex: vector = self * n
    def __mul__(self, other):
        """Multiplies the vector by a scalar and returns the result as a new vector
            Arguments:
                other -- the scalar multiplier
        """
        vector_ = Vector(self.x * other, self.y * other, self.z * other)
        return vector_

    # method : scalar division
    # ex: self.div(n)
    def div(self, n):
        """Divides the vector by a scalar quantity
            Arguments:
                n -- the scalar number used to divide the vector with
        """
        if n != 0:
            self.x = self.x / n
            self.y = self.y / n
            self.z = self.z / n

    # static : scalar division
    # ex: vector = self / n
    def __truediv__(self, other):
        """Divides the vector by a scalar number and returns the result as a new vector
            Arguments:
                other -- the scalar number to divide the vector with
        """
        vector_ = Vector(self.x / other, self.y / other, self.z / other)
        return vector_

    # static: finds distance between two points
    def dist(self, other):
        """Returns the distance between two separate points
            Argument:
                other -- the second point
        """
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        return dist

    # static : dot product
    def dot(self, other):
        """Returns the dot product of two vectors
            Arguments:
                other -- the other vector to be applied to the dot product
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    # static/method : cross product
    def cross(self, other):
        """Sets the current vector to the cross product of itself and another vector
            Arguments:
                other -- the vector that the callee is being crossed with
        """
        cross_x = self.y * other.z - other.y * self.z
        cross_y = self.z * other.x - other.z * self.x
        cross_z = self.x * other.y - other.x * self.y
        self.set(cross_x, cross_y, cross_z)

    # method : normalize
    def normalize(self):
        """Turns the callee vector into a unit vector"""
        denominator = self.mag()
        if denominator is not 0 or 1:
            self.div(denominator)

    # method : set limit
    def limit(self, limitation):
        """Limits the magnitude of the vector to a certain value
            Arguments:
                limitation -- the new limit for the callee vector
        """
        if self.mag_sq() > limitation * limitation:
            self.set_mag(limitation)

    # method : set magnitude of vector
    def set_mag(self, magnitude):
        """Sets the magnitude of the callee vector to a specific value
            Arguments:
                magnitude -- the new value for the magnitude of the callee vector
        """
        self.normalize()
        self.mult(magnitude)

    # find angle of rotation for 2D vector
    def heading(self):
        """Calculates the angle of rotation for the vector"""
        angle = math.atan2(self.y, self.x)
        return angle

    # method : rotates vector by an angle
    def rotate(self, angle):
        """Rotates the callee vector a set number of radians
            Arguments:
                angle -- the angle in radians that the vector is rotated by
        """
        x_ = self.x
        self.x = self.x * math.cos(math.radians(angle)) - self.y * math.sin(math.radians(angle))
        self.y = x_ * math.sin(math.radians(angle)) + self.y * math.cos(math.radians(angle))

    # static : finds the angle between two vectors
    @staticmethod
    def angle_between(v1, v2):
        """Determines the angle between two different vectors
            Arguements:
                v1 -- the first vector
                v2 -- the second vector
        """
        if v1.x == 0 and v1.y == 0 and v1.z == 0:
            return float(0)
        if v2.x == 0 and v2.y == 0 and v2.z == 0:
            return float(0)
        dot = v1.dot(v2)
        v1mag = v1.mag()
        v2mag = v2.mag()
        amount = dot / (v1mag * v2mag)
        if amount <= -1:
            return math.pi
        elif amount >= 1:
            return 0
        else:
            return math.acos(amount)

    def __eq__(self, other):
        """Checks to see if a vector is equal to the callee vector
            Arguments:
                other -- the vector compared to the callee
        """
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
