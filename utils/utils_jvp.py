import torch
import torch.autograd.forward_ad as fwAD


def jvp_custom(model, input, tangent):
    """function that calculates Jacobian vector product (J(model, input)tangent)"""
    with torch.no_grad():
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(input, tangent)
            out = model(dual_input)
            jvp = fwAD.unpack_dual(out).tangent
    return jvp
