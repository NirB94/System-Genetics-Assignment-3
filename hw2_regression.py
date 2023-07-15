import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from scipy.stats import f
import seaborn as sns


def count_relevant_alleles(phe, geno, ignore_hetero=True):
    """This function counts the relevant alleles: Alleles that are homozygous and heterozygous (depends on
    ignore_hetero variable).
    This function receives a phenotype Series, a genotype Series, and can also receive an
    indicator for whether to ignore heterozygous alleles or not.
    The function ignores NaN values in phe Series. The
    function returns the following:
    n - number of relevant alleles.
    snp - a dictionary of value counts for each possible allele.
    y_sum - sum of all relevant phenotype values.
    y_squared - sum of square of all relevant phenotype values.
    mul - sum of all genotype values times the corresponding phenotype values. """
    indices = phe.index.to_list()[7:]
    snp = {'B': 0, 'D': 0, 'H': 0, 'U': 0}
    weights = {'B': 0, 'D': 1 if ignore_hetero else 2, 'H': 0 if ignore_hetero else 1, 'U': 0}
    n = 0
    y_sum = 0
    y_squared = 0
    mul = 0
    for index in indices:
        curr_gen = geno[index]
        curr_phe = phe[index]

        if pd.isnull(curr_phe):
            geno[index] = None
        else:
            if ignore_hetero and curr_gen == 'H':
                phe[index] = None
                geno[index] = None
                continue
            snp[curr_gen] += 1
            n += 1
            y_sum += curr_phe
            y_squared += curr_phe ** 2
            mul += curr_phe * weights[curr_gen]
    if not ignore_hetero:
        snp['D'] = 2 * snp['D']
    return n, snp, y_sum, y_squared, mul


def calc_average(s, n):
    return s / n


def calc_sxx(x_squared, n, x_med):
    return x_squared - n * (x_med ** 2)


def calc_sxy(mulxy, n, x_med, y_med):
    return mulxy - n * x_med * y_med


def calc_y_hat(samples, beta0, beta1):
    y_hats = []
    for i in range(len(samples)):
        y_hats.append(beta0 + beta1 * samples[i])
    return y_hats


def calc_ss(ys, y_med):
    ss = 0
    for y in ys:
        ss += (y - y_med) ** 2
    return ss


def calc_ms(ss, n, k):
    return ss / (n - k)


def calc_r_squared(sse, sst):
    return 1 - (sse / sst)


def calc_f_test(msr, mse):
    return msr / mse


def regression(phe, geno, ignore_hetero=True):
    """This function models a simple linear regression.
    The function receives a phenotype Series, a genotype Series, and can also receive an indicator for whether to ignore heterozygous alleles or not.
    The function then hard-copies the Series to prevent changes in the original DataFrame.
    The function calls for count_relevant_alleles() function.
    The function then calculates all relevant values for a linear regression model.
    the function returns the P-value of the statistic F*, using f method of scipy.stats library."""
    phe = dc(phe)
    geno = dc(geno)
    n, counts, y_sum, y_squares, mulxy = count_relevant_alleles(phe, geno, ignore_hetero)
    y_average = round(calc_average(y_sum, n), 3)
    x_average = calc_average(counts['D'] + counts['H'], n)
    sum_of_x = (counts['D'] * 2 + counts['H']) if not ignore_hetero else counts['D']
    sxx = calc_sxx(sum_of_x, n, x_average)
    sxy = calc_sxy(mulxy, n, x_average, y_average)
    beta1 = sxy / sxx
    beta0 = y_average - beta1 * x_average
    y_hats = calc_y_hat(geno[4:].replace(
        {'D': (1 if ignore_hetero else 2), 'B': 0, 'U': 0, 'H': (1 if not ignore_hetero else 0)}).dropna().to_list(),
                        beta0, beta1)
    ssr = calc_ss(y_hats, y_average)
    sst = calc_ss(phe[7:].dropna(), y_average)
    sse = sst - ssr
    msr = calc_ms(ssr, 2, 1)
    mse = calc_ms(sse, n, 2)
    f_star = calc_f_test(msr, mse)
    p_val = f.sf(f_star, dfn=1, dfd=n - 2)
    return p_val


def build_p_val_df(phenotypes, genotypes, phenotype_id=1946, save=False):
    """This function calculates a simple linear regression model for each genotype in the genotypes file.
    The function receives two DataFrames: One containing the phenotypes, and the other for genotypes.
    The function can also receive a phenotype_id - the index of the chosen phenotype. Default value is my own choice - number 1946.
    The function can also receive a save indicator, whether to save the final DataFrame, or not.
    The function then runs the regression() function for each locus in the genotypes file.
    The function returns a DataFrame consisting of:
    Locus column - a column containing all loci in genotypes file.
    Chr column - the number of the chromosome in which the locus is located in.
    P_value column - containing the P-value calculated in the regression model.
    -log_P_value column - a column containing the value of the P-values after a minus log (base of 10) was preformed."""
    phenotype = phenotypes.loc[phenotype_id]
    p_vals = []
    loci = genotypes['Locus']
    chr_build37 = genotypes['Chr_Build37']
    for locus in loci:
        p_vals.append(regression(phenotype, genotypes[genotypes['Locus'] == locus].squeeze().replace({'b': 'B', 'd': 'D'})))
    p_vals = np.array(p_vals)
    minus_log_p_vals = - np.log10(p_vals)
    p_vals_df = pd.DataFrame({'Locus': loci, 'Chr': chr_build37, 'P_value': p_vals, '-log_P_value': minus_log_p_vals})
    if save:
        p_vals_df.to_excel('-log_p_value_results.xlsx')
    return p_vals_df


def plot_the_big_apple(table, save=False, show=True):
    """This function plots a Manhattan Plot for the DataFrame created in build_p_val_df() function.
    if save is True, a png of the plot is saved in the codes' directory.
    if show is True, the plot will be presented."""
    table = table.sort_values('Chr')
    table['index'] = range(len(table))
    table_grouped = table.groupby('Chr')
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    colors = sns.color_palette(None, 20)
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(table_grouped):
        group.plot(kind='scatter', x='index', y='-log_P_value', color=colors[num], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['index'].iloc[-1] + group['index'].iloc[0]) / 2)
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)

    ax.set_xlim([0, len(table)])
    ax.set_ylim([0, 4.5])

    ax.set_xlabel('Chromosome')
    plt.title('Manhattan Plot')
    if save:
        plt.savefig('Manhattan Plot.png')
    if show:
        plt.show()


def find_best_snp(table, to_print=True):
    """This function finds the best SNP i.e. the SNP with the lowest P-value.
    if to_print is True, the function prints a formal and pleasant message informing of the best SNP and its P-value."""
    best_snp_row = table.iloc[table['P_value'].idxmin()]
    best_locus = best_snp_row['Locus']
    best_p_val = best_snp_row['P_value']
    if to_print:
        print(f"Best SNP is {best_locus} with P-value of {best_p_val}")
    return best_locus, best_p_val
