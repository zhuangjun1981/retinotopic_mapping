"""
Created on Tue Jun 24 12:16:56 2014

@author: derricw

Timing decorator.  Decorate any function or class method with @timeit to
    automatically print how long it took.

See ifmain for examples.

"""

import time


def timeit(arg=1):

    if type(arg) is int:
        repeats = arg

        def wrap(f):
            def timed(*args, **kwargs):
                times = []
                for i in range(repeats):
                    ts = time.time()
                    result = f(*args, **kwargs)
                    te = time.time()
                    times.append(te-ts)
                print 'func:%r args:[%r, %r] took: %2.10f sec (mean, %i repeats)' % \
                  (f.__name__, args, kwargs, sum(times)/float(len(times)),
                   repeats)
                return result
            return timed
        return wrap
    else:
        f = arg

        def timed(*args, **kwargs):
            ts = time.time()
            result = f(*args, **kwargs)
            te = time.time()
            print 'func:%r args:[%r, %r] took: %2.10f sec' % \
              (f.__name__, args, kwargs, te-ts)
            return result
        return timed


if __name__ == '__main__':

    @timeit  # runs once
    def example_function_0(seconds):
        time.sleep(seconds)
        return

    @timeit(3)  # runs it 3 times and shows average
    def example_function_1(seconds):
        time.sleep(seconds)
        return

    example_function_0(1)

    example_function_1(1)
