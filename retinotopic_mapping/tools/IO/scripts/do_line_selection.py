from toolbox.IO.nidaq import DigitalOutput, DigitalInput

do0 = DigitalOutput("Dev2",
                    port=0,
                    lines="0",)

di0 = DigitalOutput("Dev2",
                    port=0,
                    lines=1,)

di1 = DigitalOutput("Dev2",
                    port=0,
                    lines=2,)


do0.start()
di0.start()
di1.start()

do0.writeBit(0, 1)
#print di0.read()

do0.stop()
di0.stop()
di1.stop()

do0.clear()
di0.clear()
di1.clear()