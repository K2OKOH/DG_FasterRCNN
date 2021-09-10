class Rectangle:
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        
    def __setattr__(self, name, value):
        if name == 'square':
            self.width = value
            self.height = value
        else:
            self.name = value
            # super().__setattr__(name, value)
            
            
    def getArea(self):
        return self.height * self.width
    

r = Rectangle()
r(4,5)