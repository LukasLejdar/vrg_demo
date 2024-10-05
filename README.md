# vrg demo

Cílem je obecně řešit okrajové úlohy pro systém diferenciálních rovnic

<p align="center">
  $$ \dot{y} = f\ (y)  $$
</p>
<p align="center">
  $g(\ y( a ),\ y(b)\ ) = 0$,
</p>

s jednou volnou hranicí b. Speciálně bych potom měl být schopný splnit následující zadání.

# Simulátor střelby

- vstup:
    - 2D bod umístění střelce
    - 2D bod umístění cíle
    - úsťová rychlost střely
    - aerodynamické parametry střely
- výstup:
    - úhel náměru, při kterém je cíl zasažen, pokud uvažujeme jen odpor vzduchu úměrný druhé mocnině rychlosti, tj.

<p align="center">
  $\ddot{\vec{r}} = -c \dot{r} \dot{\vec{r}} + \vec{g}$
</p>

<p align="center">
    $$\dot{r}(0) = v_{0} \ \ \ \ \  \vec{r}(t_{0}) = \vec{r}_{0}  \ \ \ \ \  \vec{r}(t_{f}) = \vec{r}_{f}$$
</p>

# Implementace
Funkce solve_fbvp implementuje řešení okrajových úloh pomocí Hermite Simpsonovy kolokační metody kontrloy residuí, podobně, jako algoritmus z odkazu [1]. Pro $D$ rovnic a $N$ kolokačních bodů se v každém kroku počítá $D(N-1)$ tzv. zbytků 

$$R = F(t_{0}, y_{0}, t_{1}, y_{1}, ..., t_{N}, y_{N})$$

které musí být všechny nulové, aby vzniklá křivka přibližně vyhovovala zadanému systému diff. rovnic. Funkce F je v tomto případě rozdíl pravé a levé strany vztahu (14) na odkaze [3], kde je možné najít i detailní odvození. Vzniká tedy $D(N-1)$ rovnic pro $D(N-2)$ vnitřních bodů, které je potřeba doplnit počtem $D$ hraničních proměnných, aby byl problém řešitelný. Protože je jedna z hranic volná, bude jednou z těchto proměnných muset být právě konečný parametr $b$. Všechny kolokační body budou 'časově' rovnoměrně rozmístěné a hledá se krok mezi nimi. V případě Simulátoru střelby je to právě simulační krok $ts$, který doplní elevační úhel θ  a konečná rychlost $[vx, vy]$. Tj. 4 proměnné pro systém 2 diferenciálních rovnic 2. řádu, který lze jednoduše přepsat na požadovaný systém 4 rovnic 1. řádu.

Celý problém je tím převedený na systém $D(N-1)$ nelinearních rovnic pro $D(N-1)$ proměnných, který se dál řeší metodou tečen. Je potřeba najít jakobián $J$ zobrazení $F$ a v nejjednodušším případě počítat 

$$R = F(\ y_{0}(0,\ bv),\ y_{1},\ ...,\  y_{N}(ts(N-1),\ bv)\ )$$

$$(y_{1},\ ...,\ y_{N-1},\ ts,\ bv)\ = (y_{1},\ ...,\ y_{N-1},\ ts,\ bv)\ -\ J^{-1}*R$$

což by mělo vést k řešení, pokud začneme s rozumným typem pro $y$, $ts$ a $bv$. Jakobián podle proměnných $bv$ bude muset doplnit uživatel.

Zvažoval jsem ještě aproximovat řešení jedním velkým polynomem místo Hermitovské interpolace, jako na odkaze [2]. Implementace by byla o něco jednodušší, ale běžněji se používá Hermitovské interpolace, která se mnohem lépe generalizuje na jiné úlohy, takže jsem zǔstal u ní. Obě metody jsou provizorně implementované v fbvp_polynom.py a fbvp_simpson.py. 


# Odkazy

- [1] Jacek Kierzenka and Lawrence F. Shampine. 2001. A BVP solver based on residual control and the Maltab PSE. ACM Trans. Math. Softw. 27, 3 (September 2001), 299–316.
- [2] Hedengren, John & Asgharzadeh Shishavan, Reza & Powell, Kody & Edgar, Thomas. (2014). Nonlinear Modeling, Estimation and Predictive Control in APMonitor. Computers & Chemical Engineering. 70. 10.1016/j.compchemeng.2014.04.013. 
        https://www.researchgate.net/publication/262191158_Nonlinear_Modeling_Estimation_and_Predictive_Control_in_APMonitor
- [3] Enric celaya (2021). Second Order Collocation
        https://www.iri.upc.edu/files/scidoc/2498-Second-order-collocation.pdf
