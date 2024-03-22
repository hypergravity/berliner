def eval_mbol(log_L):
    Mbol = 4.77 - 2.5 * log_L
    return Mbol


def eval_logg(log_mass, log_Teff, log_L):
    log_g = -10.616 + log_mass + 4.0 * log_Teff - log_L
    return log_g
