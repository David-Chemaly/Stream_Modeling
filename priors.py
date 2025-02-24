import scipy

def stream_large_prior_transform_ndim16(p):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, y0, z0, v0, dirvx, dirvy, dirvz, \
    t_end = p

    logM1  = (11 + 4*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    logm1 = (6 + 3*logm) 
    rs1   = (1 + 2*rs)

    x1, y1, z1 = [
        scipy.special.ndtri(_) * 250 for _ in [x0, y0, z0]
    ]

    v1 = 3*v0 
    dirvx1, dirvy1, dirvz1 = [
        scipy.special.ndtri(_) for _ in [dirvx, dirvy, dirvz ]
    ]

    t_end1 = 5*t_end + 1

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, y1, z1, v1, dirvx1, dirvy1, dirvz1,
            t_end1]

def stream_large_prior_transform_ndim15(p):
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, v0, dirvx, dirvy, dirvz, \
    t_end = p

    logM1  = (11 + 4*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    logm1 = (6 + 3*logm) 
    rs1   = (1 + 2*rs)

    x1, z1 = [
        scipy.special.ndtri(_) * 250 for _ in [x0, z0]
    ]

    v1 = 3*v0 
    dirvx1, dirvy1, dirvz1 = [
        scipy.special.ndtri(_) for _ in [dirvx, dirvy, dirvz ]
    ]

    t_end1 = 5*t_end + 1

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, z1, v1, dirvx1, dirvy1, dirvz1,
            t_end1]

# No time
def orbit_large_prior_transform_ndim12(p):
    logM, Rs, q, dirx, diry, dirz, \
    x0, z0, v0, dirvx, dirvy, dirvz = p

    logM1  = (11 + 3*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + 4*q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    x1, z1 = [
        scipy.special.ndtri(_) * 250 for _ in [0.5 + x0/2, z0]
    ]

    v1 = 3*v0 
    dirvx1, dirvy1, dirvz1 = [
        scipy.special.ndtri(_) for _ in [dirvx, 0.5 + dirvy/2, dirvz ]
    ]

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            x1, z1, v1, dirvx1, dirvy1, dirvz1]

def orbit_unrestricted_prior_transform_ndim13(p):
    logM, Rs, q, dirx, diry, dirz, \
    x0, y0, z0, v0, dirvx, dirvy, dirvz = p

    logM1  = (11 + 3*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    x1, y1, z1 = [
        scipy.special.ndtri(_) * 250 for _ in [x0, y0, z0]
    ]

    v1 = 3*v0 
    dirvx1, dirvy1, dirvz1 = [
        scipy.special.ndtri(_) for _ in [dirvx, dirvy, dirvz ]
    ]

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            x1, y1, z1, v1, dirvx1, dirvy1, dirvz1]