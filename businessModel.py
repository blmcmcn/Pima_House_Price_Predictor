import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

class Downpayment:

    def __init__(self, cost):
        self.cost = cost
        self.minimum = cost * 0.2

    def __check(self, x):
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def f(self, x):
        self.__check(x)
        return self.minimum + (x * self.cost)

    def dfdx(self, x):
        self.__check(x)
        return self.cost

class Loan:

    def __init__(self, sale_price, APR, years):

        # calculate some dependent variables
        closing_cost = sale_price * .03
        cost = sale_price + closing_cost

        # assign important variables as attributes
        self.cost = cost # the total initial cost of the house
        self.MPR = APR / 12 # the monthly percentage rate
        self.nper = years * 12 # number of periods in the loan
        self.downpayment = Downpayment(cost) # the downpayment

    def __check(self, x):
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def discount_factor(self):
        return (self.MPR * (1 + self.MPR)**self.nper) / ((1 + self.MPR)**self.nper - 1)

    def monthly_payment(self, x):
        self.__check(x)
        return (self.cost - self.downpayment.f(x)) * self.discount_factor()

    def f(self, x):
        return self.monthly_payment(x) * self.nper

    def dfdx(self, x):
        return - self.downpayment.dfdx(x) * self.discount_factor() * self.nper

class RoE:

    def __init__(self, predicted, sale_price, APR, years, discount_rate=(1-.03)):

        # calculate some dependent variables
        closing_cost = sale_price * .03
        cost = sale_price + closing_cost

        # assign important variables as attributes
        self.nper = years * 12
        self.discount_rate = discount_rate
        self.years = years
        self.closing_cost = closing_cost
        self.rental_rate = 0.01 * predicted
        self.predicted = predicted
        self.downpayment = Downpayment(cost)
        self.loan = Loan(sale_price, APR, years)

    def __check(self, x, t):
        assert t >= self.years*12, "we have made the basic assumption that the number of months will be greater than or equal to the lifetime of the loan"
        assert np.all([ (x >= 0) , (x <= .8) ]), "please insert a value 0 <= x <= .8"

    def profit(self, x, t):
        self.__check(x,t)
        return self.rental_rate*t + self.predicted - self.downpayment.f(x) - self.loan.f(x)

    def equity(self, x):
        return self.predicted - self.loan.f(x)

    def f(self,x,t):
        self.__check(x,t)
        return self.discount_rate**self.years * self.profit(x,t) / self.equity(x)

    def dfdx(self,x,t):

        self.__check(x,t)

        u = self.profit(x,t)
        dudx = -(self.downpayment.dfdx(x) + self.loan.dfdx(x))
        v = self.equity(x)
        dvdx = -self.loan.dfdx(x)

        return self.discount_rate**self.years *(v*dudx - u*dvdx) / v**2

    def find_best_x(self,t):

        X = np.linspace(0,.8,1000)
        Y = np.zeros_like(X)

        for i,x in enumerate(X):
            Y[i] = self.f(x,t)

        idx = Y.argmax()
        return X[idx],Y[idx]

class BusinessModel:

    def __init__(self):
        data = pd.read_csv("data/y_hat.csv")
        data['abs_error'] = np.abs(data.saleprice - data.predicted)
        data['error'] = data.saleprice - data.predicted
        data['alpha'] = data.saleprice / data.predicted
        # data['opt_x'] = 0.55 * (data.alpha - 0.625)**(2/3)
        # data = data[data.opt_x <= 0.8]

        N = data.shape[0]
        opt_x = np.zeros(N)
        roe_max = np.zeros(N)
        down = np.zeros(N)
        for i in range(N):
            obs = data.iloc[i]
            opt_x1, roe_max1, down1 = ReturnOnEquity(obs.predicted,obs.saleprice)
            opt_x[i] = opt_x1
            roe_max[i] = roe_max1
            down[i] = down1
        data['opt_x'] = opt_x
        data['RoE_max'] = roe_max
        data['down'] = down
        data['best_value'] = roe_max / down
        self.data = data

    def median_percentage_error(self, values='abs_error'):
        self.data.sort_values(values, ascending=False, inplace=True)
        ape = 100 * self.data[values] / self.data.saleprice
        idx = ape.shape[0]//2
        return ape.iloc[idx]

    def mean_percentage_error(self, values='abs_error'):
        ape = 100 * self.data[values] / self.data.saleprice
        return ape.mean()

def plot1(t=30, APR=.05, years=30):
    """
    explore  what kind of fraction we should make as a downpayment above the minimum to maximize RoE
    """

    predicted = np.array([150, 200]) * 1000
    alpha = np.array([.6, .8,  1.0, 1.2])

    nrows = predicted.shape[0]
    ncols = alpha.shape[0]
    N = int(nrows * ncols)

    fig,ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    fig.suptitle(f"Correlation Between Sale Price:Predicted and Optimum Downpayment")
    # pdb.set_trace()

    for n in range(N):

        i,j = divmod(n,ncols)

        sale_price = alpha[j] * predicted[i]
        roe = RoE(predicted[i], sale_price, APR, years)

        X = np.linspace(0,.8,101)
        Y = np.zeros_like(X)

        for k,x in enumerate(X):
            Y[k] = roe.f(x,t*12)

        idx = Y.argmax()

        # ax[i,j].set_title("x: {:0.4f}: RoE: ${:0.2f}K".format(x[idx],y[idx]/1000))
        ax[i,j].plot(X,Y, alpha=.5)
        ax[i,j].hlines(0,0,.8, color='r')
        ax[i,j].set_xlim([0,.8])
        ax[i,j].set_ylim(-Y[idx],Y[idx])
        ax[i,j].set_xticks([0,.4,.8])
        ax[i,j].set_yticks([])
        # ax[i,j].set_xlabel("x")
        ax[i,j].set_title("x: {:0.3f}, $\\alpha$: {}\nRoE: %{:0.2f}\npredicted: ${:0.2f}K".format(X[idx],alpha[j],Y[idx],predicted[i]/1000), fontsize=8)

    # plt.tight_layout()
    fig.text(0.5, 0.04, 'Fraction of Additional Downpayment', ha='center')
    fig.text(0.04, 0.5, 'Return on Equity (fraction)', va='center', rotation='vertical')
    fig.subplots_adjust(hspace=.5, wspace=.3, bottom=.15, top=.8)
    fig.savefig(f"business_model/RoE.pdf")
    plt.close(fig)

def plot2(t=30, APR=.05, years=30):
    """
    explore how the ratio of predicted value : sale price and what 'x' optimizes RoE
    """

    predicted = 150000
    alpha = np.linspace(.01,1,101)
    sale_price = predicted * alpha

    Y = np.zeros_like(sale_price)
    X = np.zeros_like(sale_price)

    x = np.linspace(.625,1,100)
    # m = .30 / (1-x.min())
    # y = m*(x-x.min())
    # y = .5*np.sqrt(x-x.min())
    y = .55 * (x-x.min())**(2/3)
    # y = p[0] * np.exp(p[1]*(x-x.min()))**(p[2])
    # y = p[0] * (x-x.min())**(p[1])

    for i,sp in enumerate(sale_price):
        roe = RoE(predicted, sp, APR, years)
        X[i],Y[i] = roe.find_best_x(t*12)

    fig,ax = plt.subplots()
    fig.suptitle('Determining the Optimal Downpayment on a Single House')
    ax.plot(alpha,X, label='simulated data')
    ax.plot(x,y, color='r', label='model')
    ax.set_xlim(0,1)
    ax.set_ylim(0,.8)
    ax.set_xlabel('$\\alpha$ [Sale Price / Predicted Value]')
    ax.set_ylabel('x [Optimal Fraction of Additional Downpayment]')
    ax.legend()
    ax.annotate(
        "x = 0.550 ($\\alpha$ - 0.625)$^{2/3}$",
        xy = (.6,.4),
        color = 'r'
        )

    fig.subplots_adjust(top=.9)
    fig.savefig("business_model/alpha_ratio_to_predict_best_x.pdf")
    plt.close(fig)

def plot3(x=.25, predicted=150000, alpha=.9, APR=.05, years=30):

    sale_price = predicted * alpha
    roe = RoE(predicted,sale_price,APR,years)

    X = (np.linspace(0,20,100) + years) * 12
    Y = np.zeros_like(X)

    for i,t in enumerate(X):
        Y[i] = roe.f(x,t)

    idx = np.abs(Y-100).argmin()

    fig,ax = plt.subplots()
    ax.plot(X,Y, label='RoE over time')
    ax.hlines(100, X.min(), X.max(), color='r', label='break even')
    ax.vlines(X[idx], Y.min(), Y.max(), color='r')
    ax.set_xlim(X.min(),X.max())
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel('Time [months]')
    ax.set_ylabel('Return on Equity [fraction]')
    ax.set_title('Return on Equity vs. Time')

    plt.legend(loc='best')
    fig.savefig("business_model/JoE_vs_time.pdf")
    plt.close('all')

def ReturnOnEquity(predicted,saleprice, APR=.06, years=30):
    roe = RoE(predicted, saleprice, APR, years)
    x,roe_max = roe.find_best_x(years*12)
    down = roe.downpayment.f(x)
    return x, roe_max, down
