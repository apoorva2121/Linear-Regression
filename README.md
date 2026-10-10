# Car price prediction — linear regression from first principles

Which features actually drive the price of a car, and how much of the price do
they explain?

Geely Auto, a Chinese manufacturer, wants to enter the American market and needs
to know what the market prices on. The brief is deliberately **inferential, not
predictive**: the goal is to identify the significant variables and read their
coefficients, not to win on accuracy.

**Dataset:** 205 cars, 26 columns, 27 brands. Prices run \$5,118 to \$45,400
(mean \$13,277).

| | R² |
|---|---|
| Single feature (`enginesize`) | 0.758 |
| Four features chosen by RFE | 0.867 |
| **Final model, with brand dummies — train** | **0.947** |
| **Final model — held-out test** | **0.84** |

> **Read the last two rows together.** Train R² 0.947 against test R² 0.84 is a
> ~0.11 gap, and that gap is the most interesting result in this repository. It
> is what adding 20-odd brand dummy variables to 164 training rows does. The
> honest headline number is **0.84**, and the "What I would do differently"
> section below says how I would close it.

---

## What this notebook does

Seventeen models, built up one variable at a time, each justified before the
next is added. Run order:

**1. Preparation**
- Drop `car_ID` (an index, not a feature).
- Extract the manufacturer from `CarName` by splitting on the first token, then
  fix the alias collisions that creates — `vw` and `vokswagen` both become
  `volkswagen`. Left alone, those would have been treated as three separate
  brands.
- Recast `symboling` as categorical: it is an insurance risk rating, so its
  numeric values are labels, not quantities.

**2. Understand before modelling**
- Correlation matrix over the 16 numeric columns.
- Scatter plots for every variable with |correlation| > 0.5 against price.
- Pair plot, and box plots for all 10 categorical variables.

What that showed:
- **Positively correlated with price:** `wheelbase`, `carlength`, `carwidth`,
  `curbweight`, `enginesize`, `boreratio`, `horsepower`.
- **Negatively correlated:** `citympg`, `highwaympg` — efficient cars are
  cheap cars.

**3. Build up, and check multicollinearity at each step**

This is the part worth reading. Model 3 (`enginesize` + `curbweight` +
`carwidth`) returned a **negative coefficient for `carwidth`** — impossible,
since wider cars are plainly more expensive. That sign flip is the classic
symptom of multicollinearity, so rather than accept the model I computed the
variance inflation factors:

| Feature | VIF |
|---|---|
| `curbweight` | 26.3 |
| `carwidth` | 15.1 |
| `enginesize` | 10.9 |

Dropping `curbweight`, the worst offender, brought the rest to **VIF 2.11** and
the sign flip disappeared. These variables are all proxies for "how big is the
car", so they carry nearly the same information.

**4. Then the overfitting trap, on purpose**

Model 14 throws in every variable: **R² 0.977, adjusted 0.963.** It looks
superb and it is meaningless — with more columns than useful signal on 164
training rows, R² only ever goes up. That is why the manual build-up stops here
and feature selection becomes automated.

**5. RFE, residuals, and a single test**
- Recursive Feature Elimination down to 10 features, then manual pruning by
  p-value and VIF to `carwidth`, `curbweight`, `enginesize`,
  `rear_enginelocation`.
- Brand dummies added (base category: `alfa-romero`).
- Residual analysis: the error terms are centred on zero and roughly normal,
  which is the assumption OLS needs to be valid.
- The test set is scored **once**, at the end: **R² 0.84.**

## What drives the price

On this data, three things:

1. **Engine size** — the single strongest predictor; alone it explains 76% of
   the variance.
2. **Car width** — standing in for overall size and class.
3. **Rear engine location** — rare, and strongly associated with premium cars.

Brand matters too, but see the caveat above: the brand dummies are most of the
train/test gap.

## Run it

```bash
git clone https://github.com/apoorva2121/Linear-Regression.git
cd Linear-Regression
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook "Car Price Prediction(Linear Regression).ipynb"
```

`CarPrice.csv` is included, so the notebook runs end to end with no setup.
`Data Dictionary - carprices.xlsx` explains every column.

Built with `pandas`, `numpy`, `seaborn`, `matplotlib`, `statsmodels` (for the
OLS summaries and VIF) and `scikit-learn` (split, scaling, RFE).

## What I would do differently

This is 2021 work, kept because the multicollinearity reasoning still stands.
The methodology has clear gaps, and naming them is more useful than quietly
fixing the numbers:

- **One split, no cross-validation.** Every number rests on a single
  `random_state=0` 80/20 split of 205 rows. With a test set of 41 cars, R² 0.84
  carries a wide error bar. Repeated K-fold would give an honest interval
  instead of a point.
- **Model selection used the training score.** Features were kept or dropped by
  looking at train R² and p-values, which is what produced the 0.947-vs-0.84
  gap. Selection belongs on a validation fold the final score never sees.
- **The brand dummies need regularisation.** ~27 brands across 164 training rows
  is too many parameters. Ridge or lasso would shrink them and probably raise
  the test score, whereas dropping them loses real signal.
- **Price should likely be modelled on a log scale.** It is right-skewed
  (\$5k–\$45k), so the residuals are probably heteroscedastic — the model will
  be worse on expensive cars. Fitting `log(price)` and back-transforming is the
  standard fix, and I never tested for it.
- **No baseline to beat.** There is no "predict the mean" reference, so 0.84 has
  no floor to be measured against. A gradient-boosted baseline would also show
  how much the linearity assumption is costing.
- **Notebook, not a pipeline.** Nothing here is reproducible by script or
  testable. I have since written that up properly — see
  [ames-house-prices-](https://github.com/apoorva2121/ames-house-prices-), which
  solves the same class of problem with a config-driven pipeline, out-of-fold
  blending and a leakage audit.

## Licence

[MIT](LICENSE). `CarPrice.csv` is the widely circulated Geely/CarPrice teaching
dataset and is included for reproducibility.
