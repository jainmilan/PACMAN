__author__ = 'milan'

import lmfit as lm
import numpy
import pandas

def residual(pars, x1, x2, y):
    vals = pars.valuesdict()
    lr =  vals['lr']
    cr =  vals['cr']
    noise = vals['noise']

    model = (lr * x1) + (cr * x2) + noise
    return model - y


def run(df):
    data_size = len(df)

    # Internal and External Temperature
    Tint = df['int_temperature'].values
    Text = df['ext_temperature'].values
    status = df['status'].values

    Ticurr = Tint[0]
    Tilast = Tint[-1]
    Tistd = numpy.std(Tint)
    Tivar = numpy.var(Tint)
    Tidiff = Tilast - Ticurr

    Tecurr = Text[0]
    Telast = Text[-1]
    Testd = numpy.std(Text)
    Tevar = numpy.var(Text)
    Tediff = Telast - Tecurr

    Tie = numpy.subtract(Text, Tint)
    Tiecurr = Tie[0]
    Tielast = Tie[-1]
    Tiestd = numpy.std(Tie)
    Tievar = numpy.var(Tie)
    Tiediff = Tielast - Tiecurr

    ocurr = numpy.sum(status)

    if data_size < 30:
        return pandas.Series({'alpha':0, 'beta':0, 'epsilon':0, 'Text':numpy.mean(Text), 'V':0, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})

    y = df.int_temperature.diff().tail(data_size-1).values
    x1 = (df.ext_temperature - df.int_temperature).head(data_size-1).values
    x2 = df.status.head(data_size-1).values

    fit_params = lm.Parameters()
    fit_params.add('lr', value=0.5)
    fit_params.add('cr', value=0.5, vary=True)
    fit_params.add('noise', value=0.5, min=0.0)

    out = lm.minimize(residual, fit_params, kws={'x1':x1, 'x2':x2, 'y':y})
    params = lm.fit_report(fit_params)

    alpha = float(params.split("\n")[1].split('+/-')[0].split(':')[1])
    beta = float(params.split("\n")[2].split('+/-')[0].split(':')[1])
    epsilon = float(params.split("\n")[3].split('+/-')[0].split(':')[1])

    return pandas.Series({'alpha':alpha, 'beta':beta, 'epsilon':epsilon, 'Text':numpy.mean(Text), 'V':1, 'size':data_size,
                              'Ticurr':Ticurr, 'Tilast':Tilast, 'Tistd':Tistd, 'Tivar':Tivar, 'Tidiff': Tidiff,
                              'Tecurr':Tecurr, 'Telast':Telast, 'Testd':Testd, 'Tevar':Tevar, 'Tediff': Tediff,
                              'Tiecurr':Tiecurr, 'Tielast':Tielast, 'Tiestd':Tiestd, 'Tievar':Tievar, 'Tiediff': Tiediff,
                              'ocurr': ocurr})
