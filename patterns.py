from einops import rearrange


# define reshape patterns
patterns_dict = {
    'image':
        {
            'reshape':
            [
                'h w c -> 1 c h w',
                '1 c h w -> h w c',
                '1 c (n2 p2) (n1 p1) -> (n2 n1) (p2 p1) c',
                '(n2 p2) (n1 p1) c -> (n2 n1) (p2 p1) c',
                '(n2 n1) (p2 p1) c -> 1 c (n2 p2) (n1 p1)',
                '(n2 n1) -> 1 n2 n1',
                '1 n2 n1 -> 1 (n2 p2) (n1 p1)',
                '1 h w c -> 1 (h w) c'
            ],
            'mode': 'bilinear'
        },
    'mesh':
        {
            'reshape':
            [
                'd h w c -> 1 c d h w',
                '1 c d h w -> d h w c',
                '1 c (n3 p3) (n2 p2) (n1 p1) -> (n3 n2 n1) (p3 p2 p1) c',
                '(n3 p3) (n2 p2) (n1 p1) c -> (n3 n2 n1) (p3 p2 p1) c',
                '(n3 n2 n1) (p3 p2 p1) c -> 1 c (n3 p3) (n2 p2) (n1 p1)',
                '(n3 n2 n1) -> 1 n3 n2 n1',
                '1 n3 n2 n1 -> 1 (n3 p3) (n2 p2) (n1 p1)',
                '1 d h w c -> 1 (d h w) c',
            ],
            'mode': 'trilinear'
        }
}


def einops_f(x, pattern, hparams=None, f=rearrange):
    """
    Apply einops operation @f on @x according to @pattern
    with keys in namespace @hparams.
    Filter out unused keys before passing to @f.
    """
    if hparams is None:
        return f(x, pattern)

    required_keys = \
        set(pattern.replace('(', '').replace(')', '').split(' '))

    return f(x, pattern, **{k: v for k, v in vars(hparams).items()
                                if k in required_keys})