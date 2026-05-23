# FAISS vs. Annoy Query Similarity Comparison

This table compares the sum of standard L2 distances for the top 20 retrieved matches between FAISS (Exact k-NN) and Annoy (Approximate ANN). The **% Difference** column indicates the accuracy loss/distance inflation of Annoy relative to the FAISS baseline (lower % difference is better, 0.00% is perfect match overlap).

| Query | FAISS L2 Sum | Annoy L2 Sum | % Difference |
| :--- | :---: | :---: | :---: |
| 1982 FIFA World Cup squads | 20.8293 | 21.1241 | 1.42% |
| 1982 football tournament players | 20.4192 | 20.5917 | 0.84% |
| 1986 FIFA World Cup squads | 20.7389 | 20.9866 | 1.19% |
| 1986 football tournament players | 20.1247 | 20.1747 | 0.25% |
| 1990 FIFA World Cup squads | 20.6774 | 20.7688 | 0.44% |
| 1990 football tournament players | 20.1812 | 20.2041 | 0.11% |
| 1994 FIFA World Cup squads | 20.7403 | 20.9238 | 0.88% |
| 1994 football tournament players | 20.1192 | 20.2243 | 0.52% |
| 1998 FIFA World Cup squads | 20.5640 | 20.7056 | 0.69% |
| 1998 football tournament players | 20.2593 | 20.3337 | 0.37% |
| 1999 FIFA World Youth Championship squads | 20.5308 | 20.6694 | 0.68% |
| 2002 FIFA World Cup squads | 20.6857 | 20.7954 | 0.53% |
| 2002 football tournament players | 19.7831 | 19.8659 | 0.42% |
| 2003 FIFA World Youth Championship squads | 20.4313 | 20.5671 | 0.67% |
| 2005 FIFA World Youth Championship squads | 20.3886 | 20.5257 | 0.67% |
| 2006 FIFA World Cup squads | 19.8947 | 20.2708 | 1.89% |
| 2006 football tournament players | 19.1611 | 19.1952 | 0.18% |
| 2007 FIFA U-20 World Cup squads | 20.9171 | 21.4940 | 2.76% |
| 2007 football tournament players | 19.5158 | 19.6021 | 0.44% |
| 2009 FIFA U-20 World Cup squads | 21.1846 | 21.6575 | 2.23% |
| 2009 football tournament players | 19.5513 | 19.5578 | 0.03% |
| 2010 FIFA World Cup squads | 20.4668 | 20.7901 | 1.58% |
| 2010 football tournament players | 19.9009 | 19.9009 | -0.00% |
| 2013–14 Illinois Fighting Illini men's basketball team | 19.7005 | 19.7720 | 0.36% |
| 2014 NCAA Men's Division I Basketball Tournament | 19.8467 | 19.8570 | 0.05% |
| American college fraternity Delta Sigma Theta | 18.3329 | 18.5838 | 1.37% |
| American college fraternity Kappa Alpha Psi | 18.6739 | 19.0782 | 2.16% |
| American college fraternity Lambda Chi Alpha | 18.6054 | 18.7552 | 0.80% |
| American college fraternity Phi Beta Sigma | 18.4478 | 18.7604 | 1.69% |
| Delta Sigma Theta chapters | 19.8129 | 20.1403 | 1.65% |
| Delta Sigma Theta college fraternity chapters | 19.5623 | 19.8787 | 1.62% |
| FIFA football cup squads from 1982 | 21.3294 | 21.5381 | 0.98% |
| FIFA football cup squads from 1986 | 21.2061 | 21.4282 | 1.05% |
| FIFA football cup squads from 1990 | 21.2858 | 21.3483 | 0.29% |
| FIFA football cup squads from 1994 | 21.2646 | 21.2949 | 0.14% |
| FIFA football cup squads from 1998 | 21.1333 | 21.2292 | 0.45% |
| FIFA football cup squads from 2002 | 21.1876 | 21.2787 | 0.43% |
| FIFA football cup squads from 2006 | 20.6164 | 20.7978 | 0.88% |
| FIFA football cup squads from 2007 | 20.8785 | 20.9950 | 0.56% |
| FIFA football cup squads from 2009 | 21.0553 | 21.1230 | 0.32% |
| FIFA football cup squads from 2010 | 21.1085 | 21.2500 | 0.67% |
| Harvard University people | 15.2973 | 15.2973 | 0.00% |
| Harvard alumni in Massachusetts | 16.8118 | 16.8567 | 0.27% |
| Illinois college basketball team players | 20.0745 | 20.5983 | 2.61% |
| Kappa Alpha Psi chapters | 18.2844 | 18.6678 | 2.10% |
| Kappa Alpha Psi college fraternity chapters | 18.9767 | 19.2093 | 1.23% |
| Lambda Chi Alpha chapters | 18.9828 | 19.1442 | 0.85% |
| Lambda Chi Alpha college fraternity chapters | 19.7431 | 19.8306 | 0.44% |
| NCAA basketball tournament brackets | 18.8579 | 18.8906 | 0.17% |
| Phi Beta Sigma chapters | 19.5654 | 19.7150 | 0.76% |
| Phi Beta Sigma college fraternity chapters | 19.6103 | 19.9002 | 1.48% |
| famous Harvard University professors | 18.0384 | 18.0384 | 0.00% |

## Summary Statistics
- **Average Distance Accuracy Loss (% Difference)**: **0.85%**
