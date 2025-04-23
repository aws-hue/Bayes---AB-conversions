import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact
import matplotlib.pyplot as plt


class BayesABReporter:
    """
    Lightweight Bayesian + Fisher reporter for two-variant experiments.
    Analyse once → get a tidy DataFrame + ready comments.

    Parameters
    ----------
    df : pd.DataFrame
        Two rows per variant (control / test) for every slice.
    metric : str
        Column with conversion rate  (ignored if successes_col given).
    size_col : str
        Column with denominator (visits, impressions …).
    successes_col : str | None
        Absolute successes column (wins over metric).
    variant_col : str
        Column with variant labels.
    control_label / test_label : str
        Values in variant_col that mark control & test.
    group_cols : list[str]
        Optional slice columns, e.g. ["country", "device"].
    alternative : {"greater","less","two-sided"}
        What counts as “better”.
    target_prob : float
        Bayesian confidence threshold (default = 0.95).
    alpha : float
        Fisher p-value threshold (default = 0.05).
    """

    # -------------------------------------------------- init
    def __init__(
        self,
        df: pd.DataFrame,
        metric: str = "cr",
        size_col: str = "n",
        successes_col: str | None = None,
        variant_col: str = "split",
        control_label: str = "control",
        test_label: str = "test",
        group_cols: list[str] | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        rvs_size: int = 100_000,
        alternative: str = "two-sided",
        target_prob: float = 0.95,
        alpha: float = 0.05,
        max_additional: int = 100_000,
        inc_step: int = 50,
    ):
        self.df = df.copy()
        self.metric = metric
        self.size_col = size_col
        self.successes_col = successes_col
        self.variant_col = variant_col
        self.control_label = control_label
        self.test_label = test_label
        self.group_cols = group_cols or []
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.rvs_size = rvs_size
        self.alternative = alternative.lower()
        if self.alternative not in {"greater", "less", "two-sided"}:
            raise ValueError("alternative must be greater / less / two-sided")
        self.target_prob = target_prob
        self.alpha = alpha
        self.max_additional = max_additional
        self.inc_step = inc_step
        self._validate()
        self.results_: pd.DataFrame | None = None

    # -------------------------------------------------- helpers
    def _validate(self):
        need = {self.variant_col, self.size_col}
        need.add(self.successes_col or self.metric)
        miss = need - set(self.df.columns)
        if miss:
            raise ValueError(f"Columns missing: {', '.join(miss)}")

    def _row_stats(self, row):
        n = int(row[self.size_col])
        s = int(row[self.successes_col]) if self.successes_col else int(round(row[self.metric] * n))
        if s > n:
            raise ValueError("successes > trials")
        return s, n

    def _post_samples(self, s, n):
        return beta(self.alpha_prior + s, self.beta_prior + n - s).rvs(self.rvs_size)

    # -------------------------------------------------- core calculation
    def _analyse_pair(self, sub: pd.DataFrame) -> dict | None:
        a = sub[sub[self.variant_col] == self.control_label]
        b = sub[sub[self.variant_col] == self.test_label]
        if a.empty or b.empty:
            return None
        s_a, n_a = self._row_stats(a.iloc[0])
        s_b, n_b = self._row_stats(b.iloc[0])

        samp_a, samp_b = self._post_samples(s_a, n_a), self._post_samples(s_b, n_b)
        prob_g, prob_l = np.mean(samp_b > samp_a), np.mean(samp_b < samp_a)

        # choose side
        if self.alternative == "greater":
            prob_diff = prob_g
        elif self.alternative == "less":
            prob_diff = prob_l
        else:
            prob_diff = max(prob_g, prob_l)

        # Fisher exact
        p_two = fisher_exact([[s_a, n_a - s_a], [s_b, n_b - s_b]])[1]

        # lift interval
        lift = (samp_b - samp_a) / samp_a
        ci_low, ci_high = np.percentile(lift, [2.5, 97.5])

        is_sig = (
            prob_diff >= self.target_prob
            and p_two <= self.alpha
            and ((ci_high < 0) or (ci_low > 0))
        )

        return dict(
            cr_control=s_a / n_a,
            cr_test=s_b / n_b,
            n_control=n_a,
            n_test=n_b,
            prob_difference=prob_diff,
            fisher_p=p_two,
            ci_low=ci_low,
            ci_high=ci_high,
            is_sufficient=is_sig,
        )

    # -------------------------------------------------- public API
    def analyze(self, sort_by: str | None = "n_control", ascending: bool = False):
        rows = []
        iterator = self.df.groupby(self.group_cols) if self.group_cols else [(None, self.df)]
        for key, g in iterator:
            res = self._analyse_pair(g)
            if not res:
                continue
            if self.group_cols:
                res.update({c: k for c, k in zip(self.group_cols, key if isinstance(key, tuple) else (key,))})
            rows.append(res)
        self.results_ = pd.DataFrame(rows)
        if sort_by and sort_by in self.results_.columns:
            self.results_ = self.results_.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        return self.results_

    def plot_convergence(self, *slice_key, max_extra=500, step=10):
        if self.results_ is None:
            raise RuntimeError("run .analyze() first")
        if self.group_cols:
            mask = np.logical_and.reduce([self.results_[col] == val for col, val in zip(self.group_cols, slice_key)])
            row = self.results_[mask].iloc[0]
            label = "_".join(map(str, slice_key))
        else:
            row = self.results_.iloc[0]
            label = "all"

        s_a, n_a = int(row["cr_control"] * row["n_control"]), row["n_control"]
        s_b, n_b = int(row["cr_test"] * row["n_test"]), row["n_test"]

        xs, lows, highs, means = [], [], [], []
        for extra in range(0, max_extra + step, step):
            samp_a = self._post_samples(s_a + int(extra * s_a / n_a), n_a + extra)
            samp_b = self._post_samples(s_b + int(extra * s_b / n_b), n_b + extra)
            lift = (samp_b - samp_a) / samp_a
            xs.append(n_a + extra)
            lows.append(np.percentile(lift, 2.5))
            highs.append(np.percentile(lift, 97.5))
            means.append(np.mean(lift))

        plt.fill_between(xs, lows, highs, alpha=0.25)
        plt.plot(xs, means)
        plt.axhline(0, ls="--")
        plt.xlabel("control sample size")
        plt.ylabel("lift  (95 % CI)")
        plt.title(f"Lift convergence — {label}")
        plt.tight_layout()
        plt.show()


