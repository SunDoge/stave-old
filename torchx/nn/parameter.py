from jax.numpy import DeviceArray


class Parameter(DeviceArray):

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()
