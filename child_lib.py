def reverse(l):
    return list(reversed(l)) if isinstance(l, list) else tuple(reversed(l))


def mask(ent_str):
    tokens = ent_str.strip().split()
    if len(tokens) == 1:
        return '[%s]' % tokens[0]
    elif len(tokens) == 2:
        assert tokens[0] == 'the', ent_str
        return '%s [%s]' % (tokens[0], tokens[1])
    else:
        assert False, ent_str
