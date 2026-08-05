def grad_global_l2_norm(grad_list):
    sq_sum = sum((g.detach() ** 2).sum() for g in grad_list if g is not None)
    return sq_sum.sqrt()


def merged_grad_norm(grad_list1, grad_list2):
    sq_sum = 0
    for a, b in zip(grad_list1, grad_list2):
        if (a is not None) & (a is not None):
            sq_sum += sum((a + b).detach() ** 2).sum()
    return sq_sum.sqrt()


def model_params_count(model, name=None):
    if name is None:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for n, p in model.named_parameters() if name in n)

