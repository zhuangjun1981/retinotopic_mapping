from toolbox.IO.nidaq import System

if __name__ == '__main__':
    
    s = System()
    print s.getDevNames()
    print s.getScales()
    print s.getTasks()
    print s.getGlobalChans()
    print s.getNIDAQVersion()