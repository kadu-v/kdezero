

is_simple_core = False


if is_simple_core:
    from kdezero.core_simple import Variable
    from kdezero.core_simple import Function
    from kdezero.core_simple import using_config
    from kdezero.core_simple import no_grad
    from kdezero.core_simple import as_array
    from kdezero.core_simple import as_variable
    from kdezero.core_simple import setup_varibale
else:
    from kdezero.core import Variable
    from kdezero.core import Parameter
    from kdezero.core import Function
    from kdezero.core import using_config
    from kdezero.core import no_grad
    from kdezero.core import as_array
    from kdezero.core import as_variable
    from kdezero.core import setup_varibale
    from kdezero.core import Config
    from kdezero.layers import Layer
    from kdezero.models import Model


setup_varibale()
