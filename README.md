<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Personalized Music Recommendation with Bayesian Matrix Factorization

> **Example project from Spring 2025.** This is a copy of [PeytonNash/adsp_32014_final](https://github.com/PeytonNash/adsp_32014_final), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

Given how often 1,892 Last.fm users played their favorite artists, can a model predict how much a user would play an artist they haven't heard, and recommend from that? The team answered with Bayesian probabilistic matrix factorization in PyMC, and chose variational inference over MCMC when sampling would not scale to the full data.

## Summary

Each user and each artist gets a five-dimensional latent vector with hierarchical Gaussian priors, and the log of a user's play count for an artist is modeled as the dot product of the two vectors plus Gaussian noise. The team first tried NUTS sampling, which only ran on a 100-by-100 subsample, then fit the full data with automatic differentiation variational inference, 20,000 iterations and 1,000 draws from the fitted approximation. On a random 20 percent of held-out user-artist pairs the posterior-mean predictions reach an RMSE of 0.92 in log units and explain 62 percent of the variance. A recommendation function ranks unheard artists by predicted plays. The appendix notebooks record the road here: a negative-binomial likelihood justified by an overdispersion check, several hierarchical-prior variants, and tuning of the prior scales and latent dimension. The write-up explains plainly why variational inference was the right tool and what it gives up, and is honest that the example recommendations are poor, which it traces to rare artists with little user overlap.

## Team and contributions

Team "GANgsters", Spring 2025.

- **Kirthi Rao** ([@kirthi-rao](https://github.com/kirthi-rao)): the final model notebook, built as the ADVI version of the matrix factorization, with its commentary; the write-up's challenges and future work.
- **Peyton Nash** ([@PeytonNash](https://github.com/PeytonNash)): the repository, the test-set evaluation and recommendation code in the final notebook, the negative-binomial ADVI exploration in the appendix, and the write-up's sections on variational inference and ADVI.
- **Apoorva Gupta** ([@apoorvagupta-12](https://github.com/apoorvagupta-12)): the exploratory analysis, preprocessing and user-item matrix, the first NUTS model and the working notes in the appendix, and the repository's organization.
- **Zoe Calianos** ([@zoecalianos](https://github.com/zoecalianos)): the model-tuning notebook in the appendix, negative-binomial and Gaussian likelihoods, prior scales and hierarchical priors, and parts of the write-up.

The project ran from May 11 to 28, 2025. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/sp25-lastfm-bayesian-recommender/commits/main).

## Getting started

There is no environment file. The notebooks need pymc, arviz, pandas, numpy, scikit-learn, matplotlib and seaborn.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install pymc arviz pandas numpy scikit-learn matplotlib seaborn jupyter
```

Download the dataset as described under [Data](#data) and unzip its files into `Data/`. Then run `Recommendation Engine.ipynb` top to bottom: exploration, preprocessing, the model, ADVI, evaluation and an example recommendation. ADVI on the full data takes a few minutes. The appendix notebooks are the team members' individual explorations and read from the same folder.

## Data

Data files are never committed to this repository. Git ignores the dataset files in `Data/`.

- **Source:** the [HetRec 2011 Last.fm dataset](https://grouplens.org/datasets/hetrec-2011/) from GroupLens: 1,892 users, 17,632 artists, 92,834 user-artist pairs with play counts, each user's 50 most-listened artists, plus tag assignments and friend relations. The dataset's own description is kept in `Data/readme.txt`.
- **Terms:** non-commercial use, with a citation to Last.fm and to the HetRec 2011 workshop, as stated in `Data/readme.txt`.
- **How to get it:** download `hetrec2011-lastfm-2k.zip` from GroupLens and unzip it into `Data/`.
- **Where to put it:** `Data/user_artists.dat`, `Data/artists.dat` and the other `.dat` files, as the notebooks expect. The appendix notebooks also read `Data/preprocessed_user_item_matrix.csv`, which Apoorva Gupta's notebook produces from the raw files.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: problem, data, modeling approaches, the final model, variational inference and ADVI, results, challenges, future work. |
| `Recommendation Engine.ipynb` | The final notebook: exploration, preprocessing, the model, ADVI, evaluation, recommendations. |
| `Appendix/AG_0518.ipynb`, `AG_notes.txt` | Exploration, the user-item matrix, a first NUTS model on a subsample, and working notes. |
| `Appendix/PN_0521.ipynb` | A negative-binomial likelihood on raw counts, chosen after an overdispersion check, fit with ADVI. |
| `Appendix/ZC_0519.ipynb` | Model tuning: negative-binomial and Gaussian likelihoods, prior scales, hierarchical priors, latent dimensions. |
| `Data/readme.txt` | The dataset's description, statistics, format and terms. |
| `image.png` | Predicted against true log play counts on the test set. |

## Method

**Model.** Play counts are log-transformed with log(1 + count). For user $u$ and artist $i$,

$$r_{ui} \sim \mathcal{N}(U_u^\top V_i,\ \sigma)$$

where the user vectors $U_u$ and artist vectors $V_i$ are five-dimensional, each drawn from a Normal with its own mean vector and scale: $\mu_U, \mu_V \sim \mathcal{N}(0, 1)$ per dimension and $\sigma_U, \sigma_V \sim \text{HalfNormal}(1)$, with the noise scale $\sigma \sim \text{HalfNormal}(1)$.

**Inference.** ADVI in PyMC for 20,000 iterations, then 1,000 draws from the fitted mean-field Gaussian approximation. The appendix shows NUTS working only on a 100-user by 100-artist subsample with 300 tuning steps and 300 draws.

**Evaluation.** A random 80/20 split of the user-artist pairs. Predictions are the dot products of the posterior-mean vectors. RMSE, MAE and $R^2$ on the held-out pairs, in log units.

**Recommendation.** For a user, predicted plays for every artist from the posterior-mean vectors, artists already heard removed, top five returned.

### What is Bayesian here

The latent vectors have priors whose means and scales are themselves learned, so the model pools information across users and across artists: a user with few plays is pulled toward the population mean vector rather than fit to noise, which is the regularization the write-up asks of a Bayesian approach. The inference is the course's variational material in practice. The posterior over tens of thousands of latent parameters is intractable, so ADVI picks the Gaussian with independent coordinates that is closest in KL divergence to it, by stochastic gradient optimization of the evidence lower bound. The write-up's account of what that buys, speed and scale, and what it costs, no posterior correlations, is accurate; mean-field approximations also tend to understate uncertainty, which matters for a recommender that wants to use it. The appendix's overdispersion check, variance against mean before choosing a negative-binomial over a Poisson likelihood, is the kind of reasoning about the likelihood the course asks for.

What the project doesn't do is use the posterior it computed. The 1,000 draws are summarized and then predictions use the means alone, so the recommendations carry no uncertainty despite the write-up's claim that they do.

## Results

From the final notebook, on the 20 percent of held-out user-artist pairs, in log(1 + plays) units.

| Metric | Value |
|---|---|
| RMSE | 0.92 |
| MAE | 0.69 |
| $R^2$ | 0.62 |

The write-up labels the 0.69 as MSE; the notebook computes MAE. The scatter of predicted against true values in `image.png` shows the expected shrinkage toward the middle, with the model underpredicting the heaviest listeners. The example recommendation for a user whose top artists are Britney Spears, Lady Gaga and Rihanna returns five artists almost nobody in the data has heard of, which the write-up explains correctly: artists heard by a handful of users get poorly constrained vectors, and a large dot product with one of them is noise, not taste.

## What to learn from this project

Course staff added this section. The modeling and the inference choice are sound and well explained; the points below are about finishing the Bayesian argument and about checking claims.

- **Use the posterior you computed.** The draws from the approximation are the project's most valuable output and go unused. Ranking by posterior mean puts poorly constrained rare artists at the top. Ranking by a lower credible bound, or penalizing predictive variance, would have fixed the example recommendation and demonstrated what Bayesian recommendation is for.
- **Measure recommendation, not just regression.** RMSE on log play counts says how well the model fits held-out counts, not whether its top five are any good. Precision at five, or a hit rate against each user's held-out artists, would have caught the problem the example shows.
- **Run the baseline you name.** The write-up describes similarity-based collaborative filtering as the standard approach and runs none of it, so there is no evidence of what the Bayesian treatment added over a popularity baseline or a plain matrix factorization.
- **Check the numbers in the write-up.** The data has 92,834 user-artist pairs, not "over 1 million listening records"; the 0.69 is MAE, not MSE.
- **Small things.** The dataset was committed to the repository; there is no environment file; a notebook keeps git configuration commands in code cells; the team name is misspelled in the write-up's title.

What the project does well: a hierarchical prior with a stated purpose; the decision to move from NUTS to ADVI made for the right reason and explained for a reader who hasn't met variational inference; an overdispersion check before choosing a likelihood; individual explorations kept as a readable appendix; and a candid account of the cold-start failure instead of a cherry-picked example.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Spring 2025. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' notebooks are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Removed the dataset files from `Data/`, about 20 MB, in line with the course's rule that data is never committed; `Data/readme.txt` stays as the dataset's documentation, and the download is described above. The files remain in the history.
- Extended `.gitignore` to cover the data files.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it.
