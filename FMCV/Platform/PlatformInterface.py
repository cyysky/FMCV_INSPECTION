import abc


class PlatformInterface(metaclass=abc.ABCMeta):
    """
    This is the interface class that only expose the method name to upper layer.
    This class doesn't implement all the methods
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'enable_platform') and
                callable(subclass.enable_platform) and
                hasattr(subclass, 'disable_platform') and
                callable(subclass.disable_platform) and
                hasattr(subclass, 'clear_error') and
                callable(subclass.clear_error) and
                hasattr(subclass, 'move_to_home_async') and
                callable(subclass.move_to_home_async) and
                hasattr(subclass, 'move_to_point_async') and
                callable(subclass.move_to_point_async) or
                NotImplemented)

    def connection_method(self):
        """This will return what hardware interface to connect the robot, eg: comport or IP"""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model name"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def get_is_connected(self) -> str:
        """Return connection status"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def get_is_enabled(self) -> str:
        """Return platform enable/disable status"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def operating_mode(self) -> str:
        """Current operating mode"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def x(self) -> float:
        """Cartesian x"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def y(self) -> float:
        """Cartesian y"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def z(self) -> float:
        """Cartesian z"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def roll(self) -> float:
        """End effector roll"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def set_feedback_callback(self, feedback_callback):
        """Set feedback callback if unable to set it at class constructor"""
        raise NotImplementedError

    @abc.abstractmethod
    def enable_platform(self) -> bool:
        """Enable platform"""
        raise NotImplementedError

    @abc.abstractmethod
    def disable_platform(self) -> bool:
        """Disable platform"""
        raise NotImplementedError

    @abc.abstractmethod
    def clear_error(self) -> bool:
        """Clear error from platform"""
        raise NotImplementedError

    @abc.abstractmethod
    def move_to_home_async(self, complete_callback):
        """Command the platform end effector back to home position"""
        raise NotImplementedError

    @abc.abstractmethod
    def move_to_point_async(self, x, y, z, complete_callback):
        """Move the end effector based on the cartesian x, y, z value"""
        raise NotImplementedError

    #TODO: status and position feedback from robot or xy-table

