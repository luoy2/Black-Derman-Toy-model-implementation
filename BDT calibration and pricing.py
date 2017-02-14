import numpy as np
import pandas as pd
import math
from openpyxl import load_workbook
from scipy.stats import norm
import scipy.optimize
import os

# Step 1: Reading data from excel file
# os.chdir("/Users/luoy2/Dropbox/Term Structure Project")
_DAYCOUNT = True

if _DAYCOUNT == True:
    input = 'data for bdt_daycount.xlsm'
    output = 'BDT_daycount_final.xlsx'
else:
    input = 'data for bdt.xlsm'
    output = 'BDT.xlsx'

data = load_workbook(input, data_only=True)['cleaned data']
writer = pd.ExcelWriter(output)
group_member = pd.DataFrame(
    ['Group Member:', 'Yikang Luo', 'Junqi Liao', 'Xinyu Peng', 'Zhao Li'])
group_member.to_excel(writer, 'Member')

# Step 2: read P(0, T) data and sigma data, and gave them the right step.
_STEPS = 96
delta_t = 1 / 12
P = {}
i = 3
j = 0
while i <= _STEPS / 3 + 3:
    P[j] = data['C'][i].value
    i += 1
    j += 3

sigma = {}
sigma_list = [val.value for val in data['E'][50:81]]
sigma = [0]
sigma.extend([sigma_list[0]] * 2)
for i in range(len(sigma_list)):
    sigma.extend([sigma_list[i]] * 3)

pi = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))
r = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))
d = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))

# linearly interpolate missing zero coupon bond price
fill_key = set(range(_STEPS)) - set(P.keys())
a = sorted(list(set(P.keys())))
exist_key = [a[i // 2] for i in range(len(a) * 2)]
i = 2
for j in fill_key:
    P[j] = math.exp((exist_key[i] - j) * np.log(P[exist_key[i - 2]]) / 3 + (
        j - exist_key[i - 2]) * np.log(P[exist_key[i]]) / 3)
    # print(j, exist_key[i], exist_key[i-2])
    i += 1

# at t = 0
r.iloc[0][0] = -np.log(P[1]) / delta_t
U = [r.iloc[0][0]]
d.iloc[0][0] = math.exp(-r.iloc[0][0] * delta_t)
pi.iloc[0][0] = 1
# at t = 1
pi.iloc[1][0] = 0.5 * pi.iloc[0][0] * d.iloc[0][0]
pi.iloc[1][1] = 0.5 * pi.iloc[0][0] * d.iloc[0][0]


def errorfn(U, round):
    sum = 0
    for j in range(round + 1):
        sum += pi.iloc[round, j] * math.exp(-U * math.exp(
            (2 * j - round) * sigma[round] * math.sqrt(delta_t)) * delta_t)
    return pow(sum - P[round + 1], 2)


# Solve for U(1) ： U（0） = r(0,0)
W = scipy.optimize.minimize_scalar(errorfn, args=(1))
U.append(W.x)
for j in range(2):
    r.iloc[1][j] = U[1] * math.exp((2 * j - 1) * sigma[1] * math.sqrt(delta_t))
    d.iloc[1][j] = math.exp(-r.iloc[1][j] * delta_t)

# from t = 2
for step in range(2, len(P) - 1):
    print(step)
    for j in range(step + 1):
        if j == 0:
            pi.iloc[step][0] = 0.5 * pi.iloc[step - 1][0] * d.iloc[step - 1][0]
        elif step == j:
            pi.iloc[step][step] = 0.5 * pi.iloc[step - 1][step - 1] * \
                                  d.iloc[step - 1][step - 1]
        else:
            pi.iloc[step][j] = 0.5 * pi.iloc[step - 1][j] * d.iloc[step - 1][j] \
                               + 0.5 * pi.iloc[step - 1][j - 1] * \
                                 d.iloc[step - 1][j - 1]

    # solve for U
    W = scipy.optimize.minimize_scalar(errorfn, args=(step), bounds=(0, 0.1),
                                       method='bounded')
    U.append(W.x)
    for j in range(step + 1):
        try:
            r.iloc[step][j] = U[step] * math.exp(
                (2 * j - step) * sigma[step] * math.sqrt(delta_t))
            d.iloc[step][j] = math.exp(-r.iloc[step][j] * delta_t)
        except OverflowError:
            print("finished tree constructing")

# Step 3: Calibrate to caplet volatilities
strike = [val.value for val in data['G'][5:36]]
maturity = [val.value for val in data['K'][5:36]]
caplet_i = [int(x) for x in np.linspace(6, 96, 31)]


# Calculate caplet price by current short rate tree and ad_tree


def get_caplet(final_node, K, shortrate_tree, ad_tree):
    cb_tree = pd.DataFrame(np.nan, index=range(final_node + 1),
                           columns=list(range(final_node + 1)))
    cb_tree.iloc[final_node] = [1.0] * (final_node + 1)
    for i in reversed(range(final_node - 3, final_node)):
        for j in reversed(range(i + 1)):
            cb_tree.iloc[i][j] = (0.5 * cb_tree.iloc[i + 1][j + 1] + (1 - 0.5)
                                  * cb_tree.iloc[i + 1][j]) / (
                                     1 + shortrate_tree.iloc[i][j] * delta_t)
    payoff_tree = cb_tree.copy()
    payoff_tree.iloc[final_node - 3] = [
        0.25 * val * max((1 / 0.25) * (1 / val - 1) - K, 0) for val in
        cb_tree.iloc[final_node - 3]]
    cap_val = [i * j for i, j in
               zip(payoff_tree.iloc[final_node - 3],
                   ad_tree.iloc[final_node - 3])]
    return np.nansum(cap_val)


caplets_price = []
for f, k in zip(caplet_i, strike):
    caplets_price.append(get_caplet(f, k, r, pi))

# Read caplet black price
caplets_price_black = [val.value for val in data['D'][50:81]]
caplet_output = [['black caplet price'] + caplets_price_black] + [
    ['BDT caplet price'] + caplets_price]
pd.DataFrame(caplet_output).transpose().to_excel(writer, 'caplet price')
# adjust volatility
r_1 = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))
pi_1 = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))
d_1 = pd.DataFrame(np.nan, index=range(_STEPS), columns=list(range(_STEPS)))
r_1.iloc[0][0] = -np.log(P[1]) / delta_t
pi_1.iloc[0][0] = 1
d_1.iloc[0][0] = math.exp(-r_1.iloc[0][0] * delta_t)
# at t = 1
pi_1.iloc[1][0] = 0.5 * pi_1.iloc[0][0] * d.iloc[0][0]
pi_1.iloc[1][1] = 0.5 * pi_1.iloc[0][0] * d.iloc[0][0]

IV_node = [0]
IV_node.extend([val - 1 for val in caplet_i])


# Solve for sigma


def adjust_sgima(sigma_to_be_solve, caplet_num):
    global r_1
    global d_1
    global pi_1
    for step in range(IV_node[caplet_num] + 1, IV_node[caplet_num + 1] + 1):
        for j in range(step + 1):
            if j == 0:
                pi_1.iloc[step][0] = 0.5 * pi_1.iloc[step - 1][0] * \
                                     d_1.iloc[step - 1][0]
            elif step == j:
                pi_1.iloc[step][step] = 0.5 * pi_1.iloc[step - 1][step - 1] * \
                                        d_1.iloc[step - 1][step - 1]
            else:
                pi_1.iloc[step][j] = 0.5 * pi_1.iloc[step - 1][j] * \
                                     d_1.iloc[step - 1][j] \
                                     + 0.5 * pi_1.iloc[step - 1][j - 1] * \
                                       d_1.iloc[step - 1][j - 1]
            r_1.iloc[step][j] = U[step] * math.exp(
                (2 * j - step) * sigma_to_be_solve * math.sqrt(delta_t))
            d_1.iloc[step][j] = math.exp(-r.iloc[step][j] * delta_t)

    return ((get_caplet(caplet_i[caplet_num], strike[caplet_num], r_1, pi_1) -
             caplets_price_black[caplet_num]) ** 2)


adjusted_vol = []
for i in range(0, len(caplets_price_black)):
    print(i)
    W = scipy.optimize.minimize_scalar(adjust_sgima, args=(i), bounds=(0, 1),
                                       method='bounded')
    adjusted_vol.append(W.x)

# Output to excel file
change = [(i - j) / i for i, j in zip(sigma_list, adjusted_vol)]
vol_output = pd.DataFrame([['BDT Volatility'] + adjusted_vol] + [
    ['Original Volatility'] + sigma_list] +
                          [['Change'] + change] + [
                              ['Average Change'] + [np.mean(change)]])
vol_output.transpose().to_excel(writer, 'new_volumn')

caplets_price = []
for f, k in zip(caplet_i, strike):
    caplets_price.append(get_caplet(f, k, r_1, pi_1))
caplet_output = [['black caplet price'] + caplets_price_black] + [
    ['BDT caplet price'] + caplets_price]
pd.DataFrame(caplet_output).transpose().to_excel(writer, 'new caplet price')


# Step 4: calculate zero coupon price based on new short_rate tree
def get_zcb(final_node, shortrate_tree, ad_tree):
    cb_tree = pd.DataFrame(np.nan, index=range(final_node + 1),
                           columns=list(range(final_node + 1)))
    cb_tree.iloc[final_node] = [1.0] * (final_node + 1)
    for i in reversed(range(final_node)):
        for j in reversed(range(i + 1)):
            cb_tree.iloc[i][j] = (0.5 * cb_tree.iloc[i + 1][j + 1] + (
                1 - 0.5)
                                  * cb_tree.iloc[i + 1][j]) / (
                                     1 + shortrate_tree.iloc[i][
                                         j] * delta_t)
    return cb_tree.iloc[0][0]


market_zcb = [val.value for val in data['C'][3:36]]
zcb_node = [int(val) for val in np.linspace(0, 96, 33)]
new_zcb = []
for f in zcb_node:
    new_zcb.extend([get_zcb(f, r_1, pi_1)])

square_error = [(i - j) ** 2 for i, j in zip(market_zcb, new_zcb)]
output_cb = pd.DataFrame([['Market ZCB:'] + market_zcb] +
                         [['BDT ZCB:'] + new_zcb] +
                         [['square error'] + square_error])
output_cb.transpose().to_excel(writer, 'error of zcb')

# Step 5: calculate swaption based on new zero cupon bond tree

print('price swaption...')
vol_list = [x / 100 for x in
            [43.13, 46.45, 48.04, 48.34, 47.98, 47.93, 48.6, 47.87, 46.93, 45.99, 49.84, 47.89, 46.39, 45.04, 44.07]]
tenor_list = [1, 2, 3, 4, 5] * 3
expiration_list = [1] * 5 + [2] * 5 + [3] * 5
result_table = [['expiration', 'tenor', 'swap rate',
                 'swap IV', 'black price', 'tree price', 'difference']]
for swap_vol, tenor, expiration in zip(vol_list, tenor_list, expiration_list):
    cb_tree_step = (expiration + tenor) * 12 + 1
    P = {}
    i = 3 + 4 * expiration
    j = expiration * 12
    while i <= 3 + 4 * (expiration + tenor):
        P[j] = data['C'][i].value
        i += 1
        j += 3

    temp_list = sorted(list(P.keys()))
    temp_list.pop(0)
    swap_rate = (P[expiration * 12] - P[12 * (expiration + tenor)]
                 ) / (0.25 * sum([P[val] for val in temp_list]))
    K = swap_rate
    swaption_blk = (0.25 * sum([P[val] for val in temp_list]) * (
        swap_rate * norm.cdf(0.5 * swap_vol * math.sqrt(expiration)) - K * norm.cdf(
            -0.5 * swap_vol * math.sqrt(expiration))))

    # using BDT to price
    coupon = {}
    i = 0
    while i <= (cb_tree_step - 1):
        coupon[i] = 0
        i += 1
    j = 12 * expiration + 3
    while j <= 12 * (expiration + tenor):
        coupon[j] = swap_rate * 0.25
        j += 3

    BDT_tree = pd.DataFrame(np.nan, index=range(cb_tree_step),
                            columns=list(range(cb_tree_step)))
    BDT_tree.iloc[cb_tree_step - 1] = [1.0 + coupon[
        cb_tree_step - 1]] * cb_tree_step
    for i in reversed(range(cb_tree_step - 1)):
        for j in reversed(range(i + 1)):
            BDT_tree.iloc[i][j] = coupon[i] + 0.5 * (BDT_tree.iloc[i + 1][j + 1] + BDT_tree.iloc[i + 1][j]) / (
                1 + r_1.iloc[i][j] * delta_t)
    CB = BDT_tree.iloc[expiration * 12]
    swaption_payoff = CB.apply(lambda x: max(1 - x, 0), 0)
    swaption_payoff_pv = swaption_payoff * pi_1.iloc[expiration * 12]
    swaption_value = np.sum(swaption_payoff_pv[:(expiration * 12 + 1)])
    result_table.append([expiration, tenor, swap_rate, swap_vol, swaption_blk, swaption_value,
                         (swaption_value - swaption_blk) / swaption_blk])
print('finished pricing swaption!')
headers = result_table.pop(0)
swaption_output = pd.DataFrame(result_table, columns=headers)
swaption_output.to_excel(writer, 'European Swaption')

# Step 6: evaluate Bermudan swaption

# 2year swaption on 3 year swap
expiration = 0
tenor = 5
cb_tree_step = (expiration + tenor) * 12 + 1
P = {}
i = 3 + 4 * expiration
j = expiration * 12

# using BDT to price
coupon = {}
i = 0
while i <= (cb_tree_step - 1):
    coupon[i] = 0
    i += 1
j = 12 * expiration + 6
while j <= 12 * (expiration + tenor):
    coupon[j] = 0.005 * 0.5
    j += 6

value_tree = pd.DataFrame(np.nan, index=range(cb_tree_step),
                          columns=list(range(cb_tree_step)))
value_tree.iloc[cb_tree_step - 1] = [1.0 + coupon[
    cb_tree_step - 1]] * cb_tree_step
for i in reversed(range(cb_tree_step - 1)):
    for j in reversed(range(i + 1)):
        value_tree.iloc[i][j] = coupon[i] + 0.5 * (value_tree.iloc[i + 1][j + 1] + value_tree.iloc[i + 1][j]) / (
            1 + r_1.iloc[i][j] * delta_t)

# writer2 = pd.ExcelWriter("bermudan.xlsx")
# value_tree.transpose().sort_index(ascending=False).to_excel(writer2, 'value_tree.csv')
# writer2.save()


# Real World Pricing
coupon = {}
i = 0
while i <= _STEPS:
    coupon[i] = 0
    i += 1
j = 6
while j < 48:
    coupon[j] = 1000 * 0.02 * 180 / 360
    j += 6
while j < 60:
    coupon[j] = 1000 * 0.0225 * 180 / 360
    j += 6

while j < 72:
    coupon[j] = 1000 * 0.025 * 180 / 360
    j += 6

while j < 84:
    coupon[j] = 1000 * 0.04 * 180 / 360
    j += 6

while j <= 96:
    coupon[j] = 1000 * 0.06 * 180 / 360
    j += 6

product_cb_tree = pd.DataFrame(np.nan, index=range(_STEPS + 1),
                               columns=list(range(_STEPS + 1)))
product_cb_tree.iloc[_STEPS] = [1000 + coupon[_STEPS]] * (_STEPS + 1)
callable_steps = [int(x) for x in np.linspace(48, 90, 8)]
for i in reversed(range(48, _STEPS)):
    if i in callable_steps:
        print(i)
        for j in reversed(range(i + 1)):
            product_cb_tree.iloc[i][j] = coupon[i] + min((0.5 *
                                                          product_cb_tree.iloc[
                                                              i + 1][
                                                              j + 1] + (
                                                              1 - 0.5)
                                                          *
                                                          product_cb_tree.iloc[
                                                              i + 1][
                                                              j]) / (
                                                             1 + r_1.iloc[i][
                                                                 j] * delta_t),
                                                         1000)
    else:
        for j in reversed(range(i + 1)):
            product_cb_tree.iloc[i][j] = coupon[i] + (0.5 *
                                                      product_cb_tree.iloc[i + 1][
                                                          j + 1] + (
                                                          1 - 0.5)
                                                      *
                                                      product_cb_tree.iloc[i + 1][
                                                          j]) / (
                                                         1 + r_1.iloc[i][
                                                             j] * delta_t)

for i in reversed(range(48)):
    for j in reversed(range(i + 1)):
        product_cb_tree.iloc[i][j] = coupon[i] + (0.5 *
                                                  product_cb_tree.iloc[i + 1][
                                                      j + 1] + (
                                                      1 - 0.5)
                                                  *
                                                  product_cb_tree.iloc[i + 1][
                                                      j]) / (
                                                     1 + r_1.iloc[i][
                                                         j] * delta_t)

product_cb_tree.transpose().sort_index(ascending=False).to_excel(writer,
                                                                 'product.csv')
r_1.transpose().sort_index(ascending=False).to_excel(writer, 'short_rate.csv')
pi_1.transpose().sort_index(ascending=False).to_excel(writer, 'ad_tree.csv')
writer.save()
print("finished task!")
