from torch import autograd


def zeroCenteredGP(x_input, D_output):

    batch_size = x_input.size(0)

    D_grad_out = autograd.grad(
        outputs=D_output.sum(),
        inputs=x_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    D_grad_out = D_grad_out.pow(2)

    assert D_grad_out.size() == x_input.size()
    zero_centered_gp = 0.5 * D_grad_out.view(batch_size, -1).sum(1).mean()

    return zero_centered_gp
