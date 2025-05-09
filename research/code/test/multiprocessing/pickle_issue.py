
import multiprocessing as mp
import os


# This is our placeholder class, all local functions will be added as it's attributes
class _LocalFunctions:
    @classmethod
    def add_functions(cls, *args):
        for function in args:
            setattr(cls, function.__name__, function)
            function.__qualname__ = cls.__qualname__ + "." + function.__name__


def calc(num1, num2, _init=False):
    # The _init parameter is to initialize all local functions outside __main__ block without actually running the
    # whole function. Basically, you shift all local function definitions to the top and add them to our
    # _LocalFunctions class. Now, if the _init parameter is True, then this means that the function call was just to
    # initialize the local functions and you SHOULD NOT do anything else. This means that after they are initialized,
    # you simply return (check below)

    def addi(num1, num2):
        print(num1 + num2)

    # Another local function you might have
    def addi2():
        print("hahahaha")

    # Add all functions to _LocalFunctions class, separating each with a comma:
    _LocalFunctions.add_functions(addi, addi2)

    # IMPORTANT: return and don't actually execute the logic of the function if _init is True!
    if _init is True:
        return

    # Beyond here is where you put the function's actual logic including any assertions, etc.
    m = mp.Process(target=addi, args=(num1, num2))
    m.start()

    print("here is main", os.getpid())
    m.join()


# All factory functions must be initialized BEFORE the "if __name__ ..." clause. If they require any parameters,
# substitute with bogus ones and make sure to put the _init parameter value as True!
calc(0, 0, _init=True)

if __name__ == "__main__":
    a = calc(5, 6)
