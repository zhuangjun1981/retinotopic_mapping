import sys
from toolbox.IO.nidaq import Device

if __name__ == '__main__':
    
    if len(sys.argv) == 1:
        dev = 'Dev1'
    else:
        dev = sys.argv[1]

    d = Device(dev)
    print d.getDOPorts()
    print d.getDIPorts()
    print d.getDOLines()
    print d.getDILines()
    print d.getAIChannels()
    print d.getAOChannels()
    print d.getCOChannels()
    print d.getCIChannels()
    print d.getTerminals()
    print d.getProductType()
