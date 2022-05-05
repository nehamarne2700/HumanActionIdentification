class inputCoordinates:
    def __init__(self,name,cord):
        self.__className=name
        self.__coordinates=cord

    @property
    def className(self):
        return self.__className

    @property
    def coordinates(self):
        return self.__coordinates

#a=customer('a1','','','','','','','','','','','','','','','','','',0,0)
#print(a.customerID)