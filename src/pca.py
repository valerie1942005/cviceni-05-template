"""Induktivni PCA (analyza hlavnich komponent) implementovana od zakladu.

PCA vybira smery s NEJVETSIM rozptylem (nejvetsi vlastni cisla kovariancni
matice) — opacny konec spektra nez spektralni shlukovani v cv4, ktere naopak
bere vlastni cisla nejmensi. Model je induktivni: ``fit`` nauci vlastni vektory
a ``transform`` je aplikuje na nova data, diky cemuz lze naucený model ulozit
a pozdeji znovu pouzit.
"""

from __future__ import annotations

import os
import numpy as np


class PCA:
    """Induktivni PCA ve stylu sklearn (nededi z niceho).

    ``fit`` z trenovacich dat nauci prumer, vlastni cisla a vlastni vektory
    (hlavni komponenty) kovariancni matice; ``transform`` timto naucenym
    pravidlem promita libovolna nova data. To je zamerny kontrast s
    transduktivnim shlukovanim z cv4, ktere zadne prenositelne pravidlo nema.

    Znamenko vlastniho vektoru je libovolne: ``eigh`` muze vratit komponentu
    i s obracenym znamenkem a obe varianty jsou spravne. Testy proto nikdy
    neporovnavaji surove vlastni vektory, ale vysvetleny rozptyl, rekonstrukcni
    chybu nebo hodnoty v absolutni hodnote.

    Atributy (naplneny az v ``fit``, do te doby ``None``):
        components_: np.ndarray tvaru (n_features, n_components_) — hlavni
            komponenty ulozene ve sloupcich.
        mean_: np.ndarray tvaru (n_features,) — prumer trenovacich dat.
        eigenvalues_: np.ndarray — vlastni cisla serazena SESTUPNE.
        explained_variance_ratio_: np.ndarray — podil vysvetleneho rozptylu
            pripadajici na jednotliva vlastni cisla (soucet = 1).
        n_components_: int — skutecny pocet ponechanych komponent.
    """

    def __init__(
            self,
            variance_threshold: float | None = 95.0,
            n_components: int | None = None,
    ) -> None:
        """Nastavi hyperparametry PCA."""
        self.variance_threshold = variance_threshold
        self.n_components = n_components

        # Atributy naucene ve fit(); pred zavolanim fit() jsou None.
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_components_: int | None = None

    def _covariance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Spocita kovariancni matici priznaku ve vektorizovane podobe."""
        assert isinstance(x, np.ndarray), "x musí být typu np.ndarray"
        assert x.ndim == 2, "x musí mít 2 rozměry (vzorky x příznaky)"
        assert x.shape[0] >= 2, "x musí obsahovat alespoň 2 vzorky"

        m = x.shape[0]
        xc = x - np.mean(x, axis=0)
        return (xc.T @ xc) / (m - 1)

    def _select_components(self, eigenvalues: np.ndarray) -> int:
        """Urci pocet komponent podle prahu kumulativniho rozptylu."""
        assert isinstance(eigenvalues, np.ndarray) and eigenvalues.ndim == 1, "eigenvalues musí být 1D np.ndarray"
        assert np.all(np.diff(eigenvalues) <= 0), "eigenvalues musí být seřazena sestupně"
        assert self.variance_threshold is not None, "variance_threshold nesmí být None"
        assert 0 < self.variance_threshold <= 100, "variance_threshold musí být v rozsahu (0, 100]"

        ratio = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(ratio) * 100

        # Vrátí index prvního prvku >= threshold, plus 1 (protože indexujeme od 0)
        return int(np.argmax(cumulative >= self.variance_threshold)) + 1

    def fit(self, x: np.ndarray) -> "PCA":
        """Nauci PCA model z trenovacich dat ``x``."""
        self.mean_ = x.mean(axis=0)

        cov = self._covariance_matrix(x)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # === Klicovy krok: razeni SESTUPNE — "nejvetsi rozptyl prvni" ===
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()

        if self.n_components is not None:
            self.n_components_ = self.n_components
        else:
            self.n_components_ = self._select_components(self.eigenvalues_)

        # Hlavni komponenty jsou vlastni vektory ve sloupcich:
        self.components_ = eigenvectors[:, : self.n_components_]
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Nauci model a rovnou vrati promitnuta data."""
        return self.fit(x).transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Promitne data ``x`` do prostoru naucenych hlavnich komponent."""
        assert self.components_ is not None, "Model ještě nebyl natrénován"
        assert x.shape[1] == self.mean_.shape[0], "Počet příznaků x neodpovídá trénovacím datům"

        return (x - self.mean_) @ self.components_

    def inverse_transform(self, x_pca: np.ndarray) -> np.ndarray:
        """Rekonstruuje data z prostoru hlavnich komponent zpet do puvodniho."""
        assert self.components_ is not None, "Model ještě nebyl natrénován"
        assert x_pca.shape[1] == self.n_components_, "Tvar X_pca neodpovídá počtu naučených komponent"

        return x_pca @ self.components_.T + self.mean_

    def save(self, path: str) -> None:
        """Ulozi naucený model do souboru ``.npz`` pomoci ``np.savez``."""
        assert self.components_ is not None, "Nelze uložit nenatrénovaný model"

        np.savez(
            path,
            components_=self.components_,
            mean_=self.mean_,
            explained_variance_ratio_=self.explained_variance_ratio_,
            eigenvalues_=self.eigenvalues_,
            n_components_=self.n_components_
        )

    @classmethod
    def load(cls, path: str) -> "PCA":
        """Nacte model ulozený metodou ``save`` a vrati hotovou instanci."""
        assert os.path.exists(path), f"Soubor {path} neexistuje"

        data = np.load(path)
        assert 'components_' in data and 'mean_' in data, "Soubor .npz neobsahuje povinná pole ('components_', 'mean_')"
        assert data['components_'].ndim == 2, "Atribut components_ musí mít 2 rozměry"

        obj = cls()
        obj.components_ = data['components_']
        obj.mean_ = data['mean_']

        # Obnova doplňujících polí pro kompletní stav modelu
        if 'explained_variance_ratio_' in data:
            obj.explained_variance_ratio_ = data['explained_variance_ratio_']
        if 'eigenvalues_' in data:
            obj.eigenvalues_ = data['eigenvalues_']
        if 'n_components_' in data:
            obj.n_components_ = int(data['n_components_'])

        return obj
