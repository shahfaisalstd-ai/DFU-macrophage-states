class Fletcher32:
    def __init__(self,*a,**k): pass
    def decode(self, buf, out=None):
        return bytes(buf)[:-4]
