# FAISS vs. Annoy Query Quality & Performance Comparison

This table compares search accuracy (sum of standard L2 distances for the top 20 retrieved matches) and search speed (execution time in milliseconds) between FAISS (Exact k-NN) and Annoy (Approximate ANN).

| Query | FAISS L2 Sum | Annoy L2 Sum | % L2 Diff | FAISS Time (ms) | Annoy Time (ms) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1982 FIFA World Cup squads | 20.8293 | 21.1241 | 1.42% | 243.346 | 36.149 | 6.73x |
| 1982 football tournament players | 20.4192 | 20.5917 | 0.84% | 240.769 | 32.875 | 7.32x |
| 1986 FIFA World Cup squads | 20.7389 | 20.9866 | 1.19% | 294.418 | 50.689 | 5.81x |
| 1986 football tournament players | 20.1247 | 20.1747 | 0.25% | 252.815 | 40.651 | 6.22x |
| 1990 FIFA World Cup squads | 20.6774 | 20.7688 | 0.44% | 245.387 | 37.135 | 6.61x |
| 1990 football tournament players | 20.1812 | 20.2041 | 0.11% | 249.762 | 31.158 | 8.02x |
| 1994 FIFA World Cup squads | 20.7403 | 20.9238 | 0.88% | 245.234 | 45.607 | 5.38x |
| 1994 football tournament players | 20.1192 | 20.2243 | 0.52% | 242.210 | 39.845 | 6.08x |
| 1998 FIFA World Cup squads | 20.5640 | 20.7056 | 0.69% | 250.550 | 57.662 | 4.35x |
| 1998 football tournament players | 20.2593 | 20.3337 | 0.37% | 325.881 | 47.172 | 6.91x |
| 1999 FIFA World Youth Championship squads | 20.5308 | 20.6694 | 0.68% | 274.625 | 33.940 | 8.09x |
| 2002 FIFA World Cup squads | 20.6857 | 20.7954 | 0.53% | 321.649 | 57.983 | 5.55x |
| 2002 football tournament players | 19.7831 | 19.8659 | 0.42% | 274.055 | 51.645 | 5.31x |
| 2003 FIFA World Youth Championship squads | 20.4313 | 20.5671 | 0.67% | 276.150 | 33.756 | 8.18x |
| 2005 FIFA World Youth Championship squads | 20.3886 | 20.5257 | 0.67% | 327.190 | 40.073 | 8.16x |
| 2006 FIFA World Cup squads | 19.8947 | 20.2708 | 1.89% | 280.281 | 66.551 | 4.21x |
| 2006 football tournament players | 19.1611 | 19.1952 | 0.18% | 246.732 | 53.546 | 4.61x |
| 2007 FIFA U-20 World Cup squads | 20.9171 | 21.4940 | 2.76% | 246.500 | 34.812 | 7.08x |
| 2007 football tournament players | 19.5158 | 19.6021 | 0.44% | 281.306 | 31.821 | 8.84x |
| 2009 FIFA U-20 World Cup squads | 21.1846 | 21.6575 | 2.23% | 326.120 | 37.472 | 8.70x |
| 2009 football tournament players | 19.5513 | 19.5578 | 0.03% | 249.729 | 32.040 | 7.79x |
| 2010 FIFA World Cup squads | 20.4668 | 20.7901 | 1.58% | 566.687 | 121.669 | 4.66x |
| 2010 football tournament players | 19.9009 | 19.9009 | -0.00% | 249.651 | 94.107 | 2.65x |
| 2013–14 Illinois Fighting Illini men's basketball team | 19.7005 | 19.7720 | 0.36% | 279.447 | 51.796 | 5.40x |
| 2014 NCAA Men's Division I Basketball Tournament | 19.8467 | 19.8570 | 0.05% | 284.611 | 32.854 | 8.66x |
| American college fraternity Delta Sigma Theta | 18.3329 | 18.5838 | 1.37% | 247.010 | 42.140 | 5.86x |
| American college fraternity Kappa Alpha Psi | 18.6739 | 19.0782 | 2.16% | 279.147 | 42.542 | 6.56x |
| American college fraternity Lambda Chi Alpha | 18.6054 | 18.7552 | 0.80% | 322.793 | 35.947 | 8.98x |
| American college fraternity Phi Beta Sigma | 18.4478 | 18.7604 | 1.69% | 326.488 | 49.710 | 6.57x |
| Delta Sigma Theta chapters | 19.8129 | 20.1403 | 1.65% | 316.375 | 47.212 | 6.70x |
| Delta Sigma Theta college fraternity chapters | 19.5623 | 19.8787 | 1.62% | 242.461 | 45.372 | 5.34x |
| FIFA football cup squads from 1982 | 21.3294 | 21.5381 | 0.98% | 315.465 | 33.573 | 9.40x |
| FIFA football cup squads from 1986 | 21.2061 | 21.4282 | 1.05% | 291.684 | 41.741 | 6.99x |
| FIFA football cup squads from 1990 | 21.2858 | 21.3483 | 0.29% | 327.904 | 35.557 | 9.22x |
| FIFA football cup squads from 1994 | 21.2646 | 21.2949 | 0.14% | 308.754 | 38.618 | 8.00x |
| FIFA football cup squads from 1998 | 21.1333 | 21.2292 | 0.45% | 260.218 | 46.830 | 5.56x |
| FIFA football cup squads from 2002 | 21.1876 | 21.2787 | 0.43% | 242.020 | 46.518 | 5.20x |
| FIFA football cup squads from 2006 | 20.6164 | 20.7978 | 0.88% | 244.837 | 54.589 | 4.49x |
| FIFA football cup squads from 2007 | 20.8785 | 20.9950 | 0.56% | 296.033 | 29.613 | 10.00x |
| FIFA football cup squads from 2009 | 21.0553 | 21.1230 | 0.32% | 246.618 | 34.868 | 7.07x |
| FIFA football cup squads from 2010 | 21.1085 | 21.2500 | 0.67% | 288.557 | 72.926 | 3.96x |
| Harvard University people | 15.2973 | 15.2973 | 0.00% | 298.643 | 69.708 | 4.28x |
| Harvard alumni in Massachusetts | 16.8118 | 16.8567 | 0.27% | 315.058 | 54.969 | 5.73x |
| Illinois college basketball team players | 20.0745 | 20.5983 | 2.61% | 244.260 | 47.927 | 5.10x |
| Kappa Alpha Psi chapters | 18.2844 | 18.6678 | 2.10% | 268.864 | 49.766 | 5.40x |
| Kappa Alpha Psi college fraternity chapters | 18.9767 | 19.2093 | 1.23% | 244.412 | 37.646 | 6.49x |
| Lambda Chi Alpha chapters | 18.9828 | 19.1442 | 0.85% | 241.138 | 28.718 | 8.40x |
| Lambda Chi Alpha college fraternity chapters | 19.7431 | 19.8306 | 0.44% | 243.351 | 26.346 | 9.24x |
| NCAA basketball tournament brackets | 18.8579 | 18.8906 | 0.17% | 298.809 | 38.351 | 7.79x |
| Phi Beta Sigma chapters | 19.5654 | 19.7150 | 0.76% | 241.086 | 68.332 | 3.53x |
| Phi Beta Sigma college fraternity chapters | 19.6103 | 19.9002 | 1.48% | 276.370 | 53.291 | 5.19x |
| famous Harvard University professors | 18.0384 | 18.0384 | 0.00% | 247.766 | 57.733 | 4.29x |

## Summary Statistics
- **Average Distance Accuracy Loss (% Difference)**: **0.85%**
- **Average FAISS Search Time**: **279.831 ms**
- **Average Annoy Search Time**: **46.607 ms**
- **Average Speedup Factor (FAISS/Annoy)**: **6.00x**
