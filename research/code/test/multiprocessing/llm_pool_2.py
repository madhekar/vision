from multiprocessing import Pool


class Test:
    def __init__(self, x):
        self.x = x
    
    @staticmethod
    def test(x):
        return x**2


    def test_apply(self, list_):
        global r
        def r(x):
            return Test.test(x + self.x)

        with Pool() as p:
            l = p.map(r, list_)

        return l



if __name__ == '__main__':
    o = Test(2)
    print(o.test_apply(range(10)))