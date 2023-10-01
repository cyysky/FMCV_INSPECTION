#import FMCV.Platform.Dobot as Dobot
#import Jaka
import functools


class CartesianPosition():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        pass

    def set_x(self, val=0.0):
        self.x = val

    def set_y(self, val=0.0):
        self.y = val

    def set_z(self, val=0.0):
        self.z = val
        
    def set_rx(self, val=0.0):
        self.rx = val

    def set_ry(self, val=0.0):
        self.ry = val

    def set_rz(self, val=0.0):
        self.rz = val

    def set_yaw(self, val=0.0):
        self.yaw = val

    def set_pitch(self, val=0.0):
        self.pitch = val

    def set_roll(self, val=0.0):
        self.roll = val

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z
        
    def get_rx(self):
        return self.rx

    def get_ry(self):
        return self.ry

    def get_rz(self):
        return self.rz

    def get_yaw(self):
        return self.yaw

    def get_pitch(self):
        return self.pitch

    def get_roll(self):
        return self.roll

class Platform:
    @staticmethod
    def platform_factory(model="", feedback_callback=None):
        """This static factory method will create and return the instance of input model"""
        if (model == "NONE"):
            return None
        elif (model == "Dobot.MG400"):
            import FMCV.Platform.Dobot.MG400 as MG400
            return MG400.MG400(feedback_callback)
        elif (model == "Jaka.MiniCobot"):
            #TODO: return Jaka.MiniCobot(feedback_callback)
            return None
        else:
            # Cannot resolve the input model name, return nothing
            return None

    @staticmethod
    def isSubstringExists(mainstring="", substrings=[]):
        if (len(substrings) > 0):
            # initiate the 1st element to False, this won't affect the final "OR" result
            available = [False]

            # Get individual results of substrings exists in mainstring or not, and make it into a list
            mapping_list = [(x in mainstring) for x in substrings]
            available.extend(mapping_list)

            # Aggregate the available list by "OR" the whole list into single result
            return functools.reduce(lambda x, y: x or y, available)
        else:
            return False

    @staticmethod
    def available_models(filters=[]):
        """This method will return the list of available models"""
        #models = ["Dobot.MG400", "Jaka.MiniCobot", "Dobot.test", "Dobot.test2", "Jaka.MiniCobot1"]
        models = ["Dobot.MG400", "Jaka.MiniCobot"]
        if(len(filters) > 0):
            available_list = [x for x in models if Platform.isSubstringExists(x, filters)]
        else:
            available_list = models

        return available_list

