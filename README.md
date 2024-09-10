# vrg demo

Cílem je obecně řešit systém diferenciálních rovnic 1. řádu s volnými hraničními podmínkami pro neznámou b
<p align="center">
  $\dot{y} = f\ (y) $
</p>
<p align="center">
  $g(\ y( a ),\ y(b)\ ) = 0$
</p>

Speciálně bych potom měl být schopný implementovat následující zadání.

# Simulátor střelby

- vstup:
    - 2D bod umístění střelce
    - 2D bod umístění cíle
    - úsťová rychlost střely
    - aerodynamické parametry střely
- výstup:
    - úhel náměru, při kterém je cíl zasažen, pokud uvažujeme jen odpor vzduchu úměrný druhé mocnině rychlosti, tj.

<p align="center">
  $\ddot{\vec{r}} = c \dot{\vec{r}}^2 - \vec{g}$
</p>

<p align="center">
    $$| \ddot{\vec{r}}(0) | = v_{0} \ \ \ \ \  \vec{r}(t_{0}) = \vec{r}_{0}  \ \ \ \ \  \vec{r}(t_{f}) = \vec{r}_{f}$$
</p>

# Implementace

Funkce solve_fbvp implementuje Hermite Simpsonovu kolokační metodu pro kontrolu resiuí volně inspirovanou z odkazu [1]. 


# Odkazy

- [1] Jacek Kierzenka and Lawrence F. Shampine. 2001. A BVP solver based on residual control and the Maltab PSE. ACM Trans. Math. Softw. 27, 3 (September 2001), 299–316.
- [2] Hedengren, John & Asgharzadeh Shishavan, Reza & Powell, Kody & Edgar, Thomas. (2014). Nonlinear Modeling, Estimation and Predictive Control in APMonitor. Computers & Chemical Engineering. 70. 10.1016/j.compchemeng.2014.04.013. 
        https://www.researchgate.net/publication/262191158_Nonlinear_Modeling_Estimation_and_Predictive_Control_in_APMonitor
- [3] Amestoy, P. R., Duff, I. S., L’Excellent, J.-Y., & Koster, J. (2001). A fully asynchronous multifrontal solver using distributed dynamic scheduling. SIAM Journal on Matrix Analysis and Applications, 23(1), 15–41.
        https://www.iri.upc.edu/files/scidoc/2498-Second-order-collocation.pdf
