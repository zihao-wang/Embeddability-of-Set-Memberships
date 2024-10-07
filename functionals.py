import torch


def lp(x, y, p=2):
    assert p > 0
    return torch.norm(x - y, p, dim=-1)

def inner_product(x, y):
    return torch.sum(x * y, dim=-1)

def cosine_similarity(x, y):
    return torch.cosine_similarity(x, y, dim=-1)

def get_functional(name):
    name = name.lower()
    if name[0] == 'l':
        p = int(name[1:])
        def _lp(x, y):
            return lp(x, y, p)
        return _lp
    elif name == 'inner_product':
        return inner_product
    elif name == 'cosine_similarity':
        return cosine_similarity
    else:
        raise ValueError(f"Unknown functional {name}")