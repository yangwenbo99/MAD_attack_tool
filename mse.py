def mse(imgx, imgy, c1=1e-6):
    d = (imgx - imgy)
    d2 = d * d
    return d2.mean()
