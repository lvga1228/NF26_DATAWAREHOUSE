#---------- Bibliothèques standard ----------#
# Importation standard sans alias
import csv
import time

# Importation sélective de classes sans alias
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Union

# Importation sélective de fonctions sans alias
from functools import reduce

#---------- Bibliothèques tierces ----------#
# Importation standard sans alias
import numpy
import pandas
import shutil

# Importation sélective sans alias
from geopy.distance import distance

# Importation sélective de classes sans alias
from geopy.geocoders import Nominatim
from geopy.location import Location
from pyspark.sql import SparkSession

# Importation sélective de modules avec alias
from pyspark.sql import functions as spark_funcs
from pyspark.sql import types as spark_types

# Importation sélective de classes avec alias
from pyspark.sql import Column as SparkColumn
from pyspark.sql import Row as SparkRow
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Window as SparkWindow

from typing import List, Optional, Tuple

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def extract_unique_values(
    sdfs: Iterable[SparkDataFrame],
    col_names: list[str],
    new_col_name: str = "value"
    ) -> Optional[SparkDataFrame]:

    """
    Retourne toutes les valeurs uniques provenant de plusieurs colonnes sélectionnées sur plusieurs DataFrames Spark.

    Parameters
    ----------
    sdfs : `Iterable[pyspark.sql.dataframe.DataFrame]`
        Itérable de DataFrames Spark.
    col_names : `list[str]`
        Liste des noms de colonnes dont on souhaite extraire les valeurs uniques.
    new_col_name : `str`, optional
        Nom de la nouvelle colonne contenant les valeurs extraites (par défaut "value").

    Returns
    -------
    retour : `pyspark.sql.dataframe.DataFrame` ou `None`
        DataFrame Spark des valeurs uniques ou `None` si vide.
    """

    # Vérification des types des paramètres
    if not isinstance(sdfs, Iterable):
        raise TypeError(f"Le paramètre 'sdfs' doit être un itérable de DataFrames Spark, trouvé {type(sdfs)}.")
    if not isinstance(col_names, list):
        raise TypeError(f"Le paramètre 'col_names' doit être une liste de chaînes, trouvé {type(col_names)}.")
    if not all(isinstance(col_name, str) for col_name in col_names):
        raise TypeError(f"Tous les éléments du paramètre 'col_names' doit être des chaînes.")
    if not isinstance(new_col_name, str):
        raise TypeError(f"Le paramètre 'new_col_name' doit être une chaîne, trouvé {type(new_col_name)}.")

    # Liste pour stocker les colonnes sélectionnées de tous les DataFrames Spark
    selected_cols: list[SparkDataFrame] = []

    # Extrait les colonnes spécifiées de chaque DataFrame Spark
    for idx, sdf in enumerate(sdfs):
        if not isinstance(sdf, SparkDataFrame):
            raise TypeError(f"L'élément {idx} du paramètre 'sdfs' n'est pas un DataFrame Spark, trouvé {type(sdf)}.")

        available_cols: list[str] = [col for col in col_names if col in sdf.columns]
        for available_col in available_cols:
            selected_cols.append(sdf.select(spark_funcs.col(available_col).alias(new_col_name)))

    if not selected_cols: return None

    # Fusionne toutes les colonnes extraites et affiche toutes leurs valeurs uniques 
    merged_values: SparkDataFrame = reduce(lambda df1, df2: df1.unionByName(df2), selected_cols)
    return merged_values.distinct()

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def check_atypicals(
    sdf: SparkDataFrame,
    atypical_type: Literal["nan", "null", "empty"],
    show_details: bool
    ) -> bool:

    """
    Vérifie la présence de valeurs atypiques (`nulles`, `NaN` ou chaînes vides) dans un DataFrame Spark.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark à analyser.
    atypical_type : `"nan" | "null" | "empty"`
        Type de valeur atypique à détecter :
            - "null" correspond aux valeurs nulles,
            - "nan" correspond aux Not-a-Number (applicable uniquement aux colonnes de type float ou double),
            - "empty" correspond aux chaînes de caractères vides.
    show_details : `bool`
        Booléen spécifiant si un rapport détaillé des colonnes concernées est affiché.

    Returns
    -------
    retour : `bool`
        Booléen indiquant si le DataFrame Spark contient des valeurs atypiques.
    """

    # Vérification que atypical_type est bien "null", "nan" ou "empty"
    if atypical_type not in ("nan", "null", "empty"):
        raise ValueError(f"Le paramètre 'atypical_type' doit être 'nan' ou 'null', valeur actuelle : '{atypical_type}'.")

    # Récupère les types de colonnes sous forme de dictionnaire : {nom_colonne: type}
    schema: dict[str, str] = dict(sdf.dtypes)

    # Liste des expressions de détection à appliquer à chaque colonne
    expressions: list[SparkColumn] = []

    # Construit les expressions conditionnelles pour chaque colonne
    for col_name in sdf.columns:
        col_type: str = schema[col_name]
        if atypical_type == "null":
            condition: SparkColumn = spark_funcs.isnull(spark_funcs.col(col_name))
        elif atypical_type == "nan":
            if col_type not in ("float", "double"): continue
            condition: SparkColumn = spark_funcs.isnan(spark_funcs.col(col_name))
        elif atypical_type == "empty":
            if not (col_type == "string"): continue
            condition = spark_funcs.trim(spark_funcs.col(col_name)) == ""
        expressions.append(spark_funcs.sum(spark_funcs.when(condition, 1).otherwise(0)).alias(col_name))

    # Si aucune expression applicable (ex : pas de colonnes float ou double pour NaN)
    if not expressions:
        if show_details: print(f"> Aucune colonne applicable pour le type '{atypical_type}'.")
        return False

    # Applique les expressions et récupère les résultats de comptage
    atypical_counts: SparkDataFrame = sdf.select(expressions)
    atypical_counts_row: SparkRow = atypical_counts.first()

    # Construit un dictionnaire des colonnes contenant des valeurs atypiques (> 0)
    atypical_info: dict[str, int] = {
        col_name: atypical_counts_row[col_name]
        for col_name in atypical_counts.columns if atypical_counts_row[col_name] > 0}

    # Affiche les détails si demandé
    if show_details:
        label: dict[str, str] = {
            "null": "nulles",
            "nan": "NaN",
            "empty": "vides"
        }[atypical_type]
        if atypical_info:
            print(f"> Les colonnes suivantes contiennent des valeurs {label} :")
            total_rows: int = sdf.count()
            for col_name, count in atypical_info.items():
                pourcentage: float = (count / total_rows) * 100 if total_rows > 0 else 0
                print(f"   - {col_name} : {count} ({pourcentage:.2f}%)")
        else: print(f"> Aucune valeur {label} détectée.")

    # Retourne True si des valeurs atypiques ont été trouvées
    return bool(atypical_info)

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def update_nulls_on_key(
    dst_sdf: SparkDataFrame,
    src_sdf: SparkDataFrame,
    key_col: str,
    target_col_in_dst: str,
    value_col_in_src: str
    ) -> SparkDataFrame:

    """
    Met à jour les valeurs nulles d'une colonne dans `dst_sdf` à l'aide des valeurs de `src_sdf`,
    en s'appuyant sur une clé de jointure.

    Parameters
    ----------
    dst_sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark principal contenant des valeurs nulles.
    src_sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark source contenant les valeurs de remplacement.
    key_col : `str`
        Nom de la colonne clé (identifiant unique).
    target_col_in_dst : `str`
        Colonne du `dst_sdf` à mettre à jour si elle est nulle.
    value_col_in_src : `str`
        Colonne dans `src_sdf` à utiliser pour la mise à jour.

    Returns
    -------
    retour : `pyspark.sql.dataframe.DataFrame`
        Nouveau DataFrame Spark avec la colonne mise à jour.
    """

    # Vérification des types des paramètres
    if not isinstance(dst_sdf, SparkDataFrame):
        raise TypeError(f"Le paramètre 'dst_sdf' doit être un Spark DataFrame, trouvé {type(dst_sdf)}.")
    if not isinstance(src_sdf, SparkDataFrame):
        raise TypeError(f"Le paramètre 'src_sdf' doit être un Spark DataFrame, trouvé {type(src_sdf)}.")
    if not isinstance(key_col, str):
        raise TypeError(f"Le paramètre 'key_col' doit être une chaîne, trouvé {type(key_col)}.")
    if not isinstance(target_col_in_dst, str):
        raise TypeError(f"Le paramètre 'target_col_in_dst' doit être une chaîne, trouvé {type(target_col_in_dst)}.")
    if not isinstance(value_col_in_src, str):
        raise TypeError(f"Le paramètre 'value_col_in_src' doit être une chaîne, trouvé {type(value_col_in_src)}.")

    value_col_renamed: str = f"{value_col_in_src}_src"
    src_sdf_minimal: SparkDataFrame = src_sdf \
        .select(
            key_col,
            spark_funcs.col(value_col_in_src).alias(value_col_renamed)) \
        .filter(spark_funcs.col(value_col_in_src).isNotNull())

    return dst_sdf \
        .join(
            other=src_sdf_minimal,
            on=key_col,
            how="left") \
        .withColumn(
            target_col_in_dst,
            spark_funcs.coalesce(
                spark_funcs.col(target_col_in_dst),
                spark_funcs.col(value_col_renamed))) \
        .select(dst_sdf.columns)

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def get_city_locations_from_sdf(
    sdf_cities: SparkDataFrame,
    geolocator: Nominatim,
    geocode_delay: float,
    allowed_type: list[str],
    ) -> tuple[dict[str, dict[str, Optional[Location]]], list[str]]:

    """
    Retourne les locations pour une liste de villes.

    Parameters
    ----------
    sdf_cities : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark contenant une seule colonne de ville, où chaque valeur est :
            - soit un nom de ville, par exemple "Paris",
            - soit un nom de ville suivi du pays, par exemple "Paris, France".
    geolocator : `geopy.geocoders.Nominatim`
        Instance Nominatim pour effectuer la géolocalisation.
    geocode_delay : `float`
        Temps d'attente (en secondes) entre deux appels à `geopy.geocoders.Nominatim.geocode()`
        pour limiter la fréquence des requêtes.
    allowed_type : `list[str]`
        Liste des types acceptés, par exemple ["administrative", "city", "village"].

    Returns
    -------
    retour[0] : `dict[str, dict[str, Optional[geopy.location.Location]]]`
        Dictionnaire de la forme {pays: {ville: objet `geopy.location.Location` ou `None`}},
        où `None` signifie que la ville a été géocodée, mais aucun résultat n'a correspondu aux types autorisés.
    retour[1] : `list[str]`
        Liste des villes pour lesquelles aucune localisation valide n'a été trouvée.
    """

    # Vérification des types des paramètres
    if not isinstance(sdf_cities, SparkDataFrame):
        raise TypeError(f"Le paramètre 'sdf_cities' doit être un Spark DataFrame, trouvé {type(sdf_cities)}.")

    # Vérification de la cohérence des paramètres
    if (len(sdf_cities.columns) != 1):
        raise ValueError(f"Le paramètre 'sdf_cities' doit être un DataFrame avec exactement une colonne, trouvé {len(sdf_cities.columns)}.")
    if sdf_cities.dtypes[0][1].lower() != "string":
        raise TypeError(f"Le type de la colonne {sdf_cities.columns[0]} du paramètre 'sdf_cities' doit être 'string', trouvé {sdf_cities.dtypes[0][1]}.")
    if sdf_cities.filter(sdf_cities[sdf_cities.columns[0]].isNull()).count():
        raise ValueError(f"Le paramètre 'sdf_cities' contient des valeurs nulles dans la colonne '{sdf_cities.columns[0]}'.")

    # Vérification que les valeurs ne sont pas vides et respectent le format attendu
    invalid_vals = [
        row[0] for row in sdf_cities.select(sdf_cities.columns[0]).collect()
        if (not row[0].strip()) or (row[0].count(",") > 1)]
    if invalid_vals:
        raise ValueError(
            f"Le paramètre 'sdf_cities' contient des valeurs invalides. "
            f"Chaque valeur doit être une chaîne non vide représentant soit une ville, soit une ville suivie d'un pays. "
            f"Exemples invalides détectés : {invalid_vals[:3]}.")

    # Extraction des villes uniques
    cities: list[str] = [row[0] for row in sdf_cities.select(sdf_cities.columns[0]).distinct().collect()]

    # Initialisation des structures de stockage des résultats
    locations: defaultdict[str, dict[str, Optional[Location]]] = defaultdict(dict)
    not_found_cities: list[str] = []

    # Géocodage ville par ville avec filtrage par type autorisé
    for city in cities:
        time.sleep(geocode_delay)
        try:
            # Requête de géocodage pour la ville actuelle
            results: Optional[list[Location]] = geolocator.geocode(city, language="en", exactly_one=False, timeout=10)
            if not results:
                not_found_cities.append(city)
                continue

            # Parcours des résultats retournés pour trouver un type accepté
            for location in results:
                if "," in city:
                    country: str = city.split(",")[1].strip()
                else:
                    country: str = location.raw.get("display_name", "").split(", ")[-1].strip()
                city: str = city.split(",")[0].strip()
                location_type: str = location.raw.get("type", "").lower()
                if location_type in allowed_type:
                    locations[country][city] = location
                    break
                else: locations[country][city] = None

        except Exception as e:
            print(f"Erreur lors de la géolocalisation de '{city}': {e}")

    # Retour des résultats (locations et villes non trouvées)
    return dict(locations), not_found_cities

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def locations_to_coords(
    spark: SparkSession,
    locations: dict[str, dict[str, Location]]
    ) -> SparkDataFrame:

    """
    Convertit un dictionnaire de localisations géographiques en un DataFrame Spark structuré.

    Parameters
    ----------
    spark : `pyspark.sql.session.SparkSession`
        Session Spark utilisée pour créer le DataFrame Spark.
    locations : `dict[str, dict[str, geopy.location.Location]]`
        Dictionnaire de la forme {pays: {ville: objet `geopy.location.Location`}}.

    Returns
    -------
    retour : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark contenant les colonnes 'Country', 'City', 'Latitude' et 'Longitude'.
    """

    rows: list[SparkRow] = []
    for country, cities in locations.items():
        for city, location in cities.items():
            rows.append(SparkRow(Country=country, City=city, Latitude=location.latitude, Longitude=location.longitude))
    sdf_coords: SparkDataFrame = spark.createDataFrame(rows)
    return sdf_coords

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def save_sdf_as_parquet(
    sdf: SparkDataFrame,
    target_path: Path,
    overwrite: bool = False,
    ) -> None:

    """
    Sauvegarde un DataFrame Spark dans le répertoire spécifié sou format Parquet.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark à sauvegarder.
    target_path : `str`
        Chemin où enregistrer le DataFrame Spark.
    overwrite : `bool`, optional
        Booléen spécifiant s'il faut écraser les données existantes (par défaut False).
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        if overwrite:
            print(f"Le chemin '{target_path}' existe déjà. Écrasement en cours...")
            sdf.write.mode("overwrite").parquet(str(target_path))
        else:
            print(f"Le chemin {target_path} existe déjà. Sauvegarde ignorée.")
    else:
        print(f"Sauvegarde du nouveau fichier vers '{target_path}'.")
        sdf.write.mode("overwrite").parquet(str(target_path))

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def create_compute_distance_udf(
    sdf_city_coords: SparkDataFrame
    ) -> Callable[[pandas.Series, pandas.Series], pandas.Series]:

    """
    Crée une UDF pour calculer la distance entre deux villes basées sur leurs coordonnées.

    Parameters
    ----------
    sdf_city_coords : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark contenant les colonnes 'Country', 'City', 'Latitude' et 'Longitude'.

    Returns
    -------
    retour : `Callable[[pandas.Series, pandas.Series], pandas.Series]`
        Pandas User-Defined Function qui prend deux colonnes (ville départ, ville destination) et retourne les distances en kilomètres.
    """

    city_coords_dict: dict[str, tuple[float, float]] = {
        row["City"]: (row["Latitude"], row["Longitude"])
        for row in sdf_city_coords.collect()}

    @spark_funcs.pandas_udf(spark_types.DoubleType())
    def compute_distance(departure_series: pandas.Series, destination_series: pandas.Series) -> pandas.Series:
        distances: List[Optional[float]] = []
        for departure, destination in zip(departure_series, destination_series):
            dep_coord: Optional[Tuple[float, float]] = city_coords_dict.get(departure)
            dst_coord: Optional[Tuple[float, float]] = city_coords_dict.get(destination)
            if dep_coord and dst_coord:
                dist: float = distance(dep_coord, dst_coord).km
                distances.append(dist)
            else:
                distances.append(None)
        return pandas.Series(distances)

    return compute_distance

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def print_max_val(
    sdf: SparkDataFrame,
    target_col: str,
    condition_cols: Iterable[str] = None,
    filter_vals: Iterable[tuple[object]] = None
    ) -> None:

    """
    Affiche la valeur maximale de `target_col` dans le DataFrame Spark.
    Si `condition_cols` et `filter_vals` sont fournis, affiche la valeur maximale
    pour chaque combinaison de conditions.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark à traiter.
    target_col : `str`
        Colonne sur laquelle on calcule le maximum.
    condition_cols : `Iterable[str]`, optional
        Colonnes servant de filtres conditionnels.
    filter_vals : `Iterable[tuple[object]]`, optional
        Itérable de tuples contenant les valeurs de filtrage.
        Chaque tuple doit correspondre à `condition_cols`.
    """

    # Vérification des types des paramètres
    if not isinstance(sdf, SparkDataFrame):
        raise TypeError(f"Le paramètre 'sdf' doit être un Spark DataFrame, trouvé {type(sdf)}.")
    if not isinstance(target_col, str):
        raise TypeError(f"Le paramètre 'target_col' doit être une chaîne, trouvé {type(target_col)}.")
    if target_col not in sdf.columns:
        raise ValueError(f"La colonne '{target_col}' n'existe pas dans le paramètre 'sdf'.")

    # Cas sans filtres : maximum global
    if (condition_cols is None) and (filter_vals is None):
        result: list[SparkRow] = sdf.select(spark_funcs.max(spark_funcs.col(target_col))).collect()
        max_val = result[0][0] if result else None
        if max_val is None: print(f"Aucune valeur trouvée dans la colonne '{target_col}'.")
        else: print(f"Valeur maximale dans la colonne '{target_col}' : {max_val}.")
        return

    # Vérification de la cohérence des paramètres
    if (condition_cols is None) or (filter_vals is None):
        raise ValueError("Les deux paramètres 'condition_cols' et 'filter_vals' doivent être fournis ensemble.")
    if (not isinstance(condition_cols, Iterable)) or isinstance(condition_cols, (str, bytes)):
        raise TypeError("Le paramètre 'condition_cols' doit être un itérable de chaînes.")
    if not all(isinstance(condition_col, str) for condition_col in condition_cols):
        raise TypeError("Tous les éléments du paramètre 'condition_cols' doivent être des chaînes de caractères.")
    if not isinstance(filter_vals, Iterable):
        raise TypeError("Le paramètre 'filter_vals' doit être un itérable de tuples.")
    if not all(isinstance(t, tuple) and (len(t) == len(condition_cols)) for t in filter_vals):
        raise ValueError("Chaque élément du paramètre 'filter_vals' doit être un tuple de même longueur que le paramètre 'condition_cols'.")

    # Parcours des conditions
    for val_tuple in filter_vals:
        filtered_df: SparkDataFrame = sdf   # Copie du DataFrame original pour appliquer les filtres successifs
        condition_strs: list[str] = []      # Pour stocker les descriptions textuelles des conditions (à afficher)

        # Applique chaque condition (colonne == valeur) du tuple
        for col_name, val in zip(condition_cols, val_tuple):
            # Vérification que la colonne de condition existe
            if col_name not in sdf.columns:
                raise ValueError(f"La colonne de condition '{col_name}' n'existe pas dans le paramètre 'sdf'.")

            # Ajoute un filtre au DataFrame pour cette condition
            filtered_df = filtered_df.filter(spark_funcs.col(col_name) == val)

            # Construit une chaîne descriptive pour cette condition
            condition_strs.append(f"{col_name} == {val}")

        # Calcule la valeur maximale dans la colonne cible pour ce sous-ensemble
        result: list[SparkRow] = filtered_df.select(spark_funcs.max(spark_funcs.col(target_col))).collect()
        max_val = result[0][0] if result else None  # Récupérer la valeur (ou None si vide)

        # Prépare une description lisible de la condition appliquée
        condition_summary = " et ".join(condition_strs)

        # Affiche le résultat
        if max_val is None: print(f"Aucune donnée pour {condition_summary}.")
        else: print(f"Valeur maximale de '{target_col}' lorsque {condition_summary} : {max_val}.")

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def apply_conditional_operations(
    sdf: SparkDataFrame,
    condition_col: str,
    target_col: str,
    rules: list[tuple[str, Literal["add", "sub", "mul", "div"], Union[int, float]]],
    output_col: str = "ajusted_value"
    ) -> SparkDataFrame:

    """
    Applique des opérations conditionnelles (addition, soustraction, multiplication, division)
    sur une colonne cible en fonction des valeurs d'une colonne de condition, et crée une nouvelle colonne en sortie.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark d'entrée.
    condition_col : `str`
        Nom de la colonne contenant les conditions, par exemple "Transport".
    target_col : `str`
        Nom de la colonne sur laquelle appliquer les opérations, par exemple "Distance".
    rules : `list[tuple[str, "add" | "sub" | "mul" | "div", float]]`
        Liste de règles de la forme [("Taxi", "mul", 1.3), ("Plane", "add", 95)].
    output_col : `str`, optional
        Nom de la colonne de sortie (par défaut "ajusted_value").

    Returns
    --------
    retour : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark avec une nouvelle colonne contenant les valeurs ajustées.
    """

    # Vérification des types des paramètres
    if not isinstance(sdf, SparkDataFrame):
        raise TypeError(f"Le paramètre 'sdf' doit être un DataFrame Spark, trouvé {type(sdf)}.")
    if not isinstance(condition_col, str):
        raise TypeError(f"Le paramètre 'condition_col' doit être une chaîne de caractères, trouvé {type(condition_col)}.")
    if not isinstance(target_col, str):
        raise TypeError(f"Le paramètre 'target_col' doit être une chaîne de caractères, trouvé {type(target_col)}.")
    if not isinstance(rules, list):
        raise TypeError(f"Le paramètre 'rules' doit être une liste, trouvé {type(rules)}.")
    if not rules:
        raise TypeError(f"Le paramètre 'rules' doit être non vide.")
    if not isinstance(output_col, str):
        raise TypeError(f"Le paramètre 'output_col' doit être une chaîne de caractères, trouvé {type(output_col)}.")

    # Vérification que chaque règle est un tuple valide
    for rule in rules:
        if (not isinstance(rule, tuple)) or (len(rule) != 3):
            raise TypeError("Chaque élément du paramètre 'rules' doit être un tuple (valeur_condition, opération, facteur).")
        condition_value, op, factor = rule
        if not isinstance(condition_value, str):
            raise TypeError(f"La valeur de condition doit être une chaîne de caractères, trouvé {type(condition_value)}.")
        if op not in ("add", "sub", "mul", "div"):
            raise ValueError(f"Opérateur non supporté : {op}. Seuls 'add', 'sub', 'mul' et 'div' sont acceptés.")
        if not isinstance(factor, (int, float)):
            raise TypeError(f"Le facteur doit être un nombre (int ou float), trouvé {type(factor)}.")

    # Fonction auxiliaire pour appliquer l'opération
    def apply_operation(col, op: str, factor):
        try:
            if op == "add":
                return col + factor
            elif op == "sub":
                return col - factor
            elif op == "mul":
                return col * factor
            elif op == "div":
                if factor == 0:
                    raise ZeroDivisionError("Division par zéro.")
                return col / factor
            else:
                raise ValueError(f"Opérateur non reconnu : '{op}'.")
        except TypeError:
            raise TypeError("Les types de 'col' et 'factor' ne sont pas compatibles avec l'opération.")

    # Construction d'une seule chaîne .when(...).when(...).otherwise(...)
    expr = spark_funcs.when(
        spark_funcs.col(condition_col) == rules[0][0],
        apply_operation(spark_funcs.col(target_col), rules[0][1], rules[0][2]))

    for condition_value, op, factor in rules[1:]:
        expr = expr.when(
            spark_funcs.col(condition_col) == condition_value,
            apply_operation(spark_funcs.col(target_col), op, factor))

    # Si aucune condition ne correspond : garder la valeur d'origine
    expr = expr.otherwise(spark_funcs.col(target_col))

    return sdf.withColumn(output_col, expr)

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def save_sdf_as_csv(
    sdf: SparkDataFrame,
    output_file: Path,
    delimiter: str = ";",
    overwrite: bool = True
    ) -> None:

    """
    Sauvegarde un DataFrame Spark dans un fichier CSV unique avec en-tête.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark à sauvegarder.
    output_file : `pathlib.Path`
        Chemin complet du fichier CSV final (ex: "./data/saved_data.csv").
    delimiter : `str`, optional
        Délimiteur CSV (par défaut: ";").
    overwrite : `bool`, optional
        Si True, écrase le fichier existant. Sinon, lève une erreur.
    """

    temp_dir: Path = output_file.parent / "__tmp_csv__"

    # Choix du mode
    mode: str = "overwrite" if overwrite else "error"

    # Écriture dans un répertoire temporaire
    sdf.coalesce(1) \
        .write \
        .option("header", True) \
        .option("delimiter", delimiter) \
        .mode(mode) \
        .csv(str(temp_dir))

    # Récupération du fichier CSV écrit par Spark
    for file in temp_dir.iterdir():
        if file.name.startswith("part-") and (file.suffix == ".csv"):
            file.replace(output_file)
            break

    # Nettoyage du dossier temporaire
    shutil.rmtree(temp_dir)

#────────────────────────────────────────────────────────────────────────────────────────────────────#

def update_csv_from_sdf(
    sdf: SparkDataFrame,
    target_file: Path,
    delimiter: str,
    deduplicate_on: Iterable[Union[str, tuple[str, ...]]] = (),
    create_if_not_exists: bool = False
    ) -> None:

    """
    Met à jour un fichier CSV unique en y ajoutant les lignes d'un DataFrame Spark.

    Parameters
    ----------
    sdf : `pyspark.sql.dataframe.DataFrame`
        DataFrame Spark à ajouter au fichier CSV.
    target_file : `pathlib.Path`
        Chemin complet du fichier CSV cible (ex: './data/updated_data.csv').
    delimiter : `str`
        Délimiteur utilisé pour écrire les données CSV (ex: ";" et ",").
    deduplicate_on : `Iterable[str | tuple[str, ...]]`
        Itérable contenant les noms de champs ou combinaisons de champs à vérifier pour la déduplication.
        Si un champ ou une combinaison de champs existe déjà dans `target_file`, la ligne correspondante n'est pas écrite.
    create_if_not_exists : `bool`, optional
        Si True, crée le fichier s'il n'existe pas encore. Sinon, lève une erreur.
    """

    if not target_file.exists():
        if not create_if_not_exists:
            raise ValueError(f"Le fichier {target_file} n'existe pas.")
        else:
            save_sdf_as_csv(sdf, target_file, delimiter=delimiter, overwrite=True)
            return

    # Si deduplicate_on est spécifié, filtrer les lignes à ajouter
    if deduplicate_on:
        spark: SparkSession = sdf.sparkSession
        sdf_target: SparkDataFrame = (spark.read
            .option("header", True)
            .option("delimiter", delimiter)
            .csv(str(target_file)))

        # Ne garder que les colonnes communes aux deux DataFrames
        sdf = sdf.select(sdf_target.columns)

        for col_spec in deduplicate_on:
            if isinstance(col_spec, str):
                sdf = sdf.join(sdf_target.select(col_spec).distinct(), on=col_spec, how="anti")
            else:
                sdf = sdf.join(sdf_target.select(*col_spec).distinct(), on=list(col_spec), how="anti")

    # Si sdf est vide après déduplication, rien à faire
    if sdf.rdd.isEmpty(): return

    temp_dir: Path = target_file.parent / "__tmp_csv_to_append__"
    sdf.coalesce(1) \
        .write \
        .option("header", False) \
        .option("delimiter", delimiter) \
        .mode("overwrite") \
        .csv(str(temp_dir))

    for file in temp_dir.iterdir():
        if file.name.startswith("part-") and (file.suffix == ".csv"):
            with (file.open("r", encoding="utf-8") as nf,
                  target_file.open("a", encoding="utf-8") as tf):
                tf.writelines(nf.readlines())
            break

    shutil.rmtree(temp_dir)

#────────────────────────────────────────────────────────────────────────────────────────────────────#

# Fonction pour les question
def get_distance_from_FCT_mission(
    sdf_FCT_mission: SparkDataFrame,
    sdf_city_coords: SparkDataFrame,
    sdf_DIM_mission_location: SparkDataFrame,
    sdf_DIM_transport: SparkDataFrame,
    min_distance_threshold: Optional[float]
    ) -> SparkDataFrame:

    distance_udf = create_compute_distance_udf(sdf_city_coords)

    sdf_mission_distance: SparkDataFrame = sdf_FCT_mission.alias("m") \
        .join(
            sdf_DIM_mission_location.alias("dpt"),
            on=spark_funcs.col("m.KEY_DEPART") == spark_funcs.col("dpt.KEY_LOCALISATION_MISSION"),
            how="inner") \
        .join(
            sdf_DIM_mission_location.alias("dst"),
            on=spark_funcs.col("m.KEY_DESTINATION") == spark_funcs.col("dst.KEY_LOCALISATION_MISSION"),
            how="inner") \
        .join(
            sdf_DIM_transport.alias("tsp"),
            on=spark_funcs.col("m.KEY_TRANSPORT") == spark_funcs.col("tsp.KEY_TRANSPORT"),
            how="inner") \
        .select(
            spark_funcs.col("m.*"),
            spark_funcs.col("dpt.VILLE").alias("VILLE_DEPART"),
            spark_funcs.col("dpt.PAYS").alias("PAYS_DEPART"),
            spark_funcs.col("dst.VILLE").alias("VILLE_DESTINATION"),
            spark_funcs.col("dst.PAYS").alias("PAYS_DESTINATION"),
            spark_funcs.col("tsp.KEY_TRANSPORT").alias("TRANSPORT")) \
        .withColumn(
            "DISTANCE_KM",
            spark_funcs.when(
                spark_funcs.col("ALLER_RETOUR") == "oui",
                distance_udf(
                    spark_funcs.col("VILLE_DEPART"),
                    spark_funcs.col("VILLE_DESTINATION")) * 2) \
            .otherwise(
                distance_udf(
                    spark_funcs.col("VILLE_DEPART"),
                    spark_funcs.col("VILLE_DESTINATION"))))

    if min_distance_threshold:
        sdf_mission_distance = sdf_mission_distance \
            .withColumn(
                "DISTANCE_KM",
                spark_funcs.when(
                    spark_funcs.col("DISTANCE_KM") < min_distance_threshold,
                    value=min_distance_threshold)
                .otherwise(spark_funcs.col("DISTANCE_KM")))

    return sdf_mission_distance

def adjust_mission_distance(
    sdf_mission_distance: SparkDataFrame,
    distance_adjustment_rules: List[Tuple[str, Literal['add', 'sub', 'mul', 'div'], Union[int, float]]]
) -> SparkDataFrame:

    return apply_conditional_operations(
        sdf=sdf_mission_distance,
        condition_col="TRANSPORT",
        target_col="DISTANCE_KM",
        rules=distance_adjustment_rules,
        output_col="AJUSTED_DISTANCE_KM"
    )

def get_mission_impact_T(
    sdf_ajusted_mission_distance: SparkDataFrame
    ) -> SparkDataFrame:

    ef_plane_shr: float = 0.2586 / 1000
    ef_plane_mid: float = 0.1875 / 1000
    ef_plane_lng: float = 0.1520 / 1000
    ef_train_shr: float = 0.0180 / 1000
    ef_train_tgv: float = 0.0033 / 1000
    ef_taxi: float = 0.2156 / 1000
    ef_bus: float = 0.1290 / 1000

    return sdf_ajusted_mission_distance \
        .withColumn(
            "EMISSIONS_T_CO2",
            spark_funcs.when(
                condition=(
                    spark_funcs.col("TRANSPORT") == "Plane") &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") < 1000),
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_plane_shr)
            .when(
                condition=(
                    spark_funcs.col("TRANSPORT") == "Plane") &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") >= 1000) &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") <= 3500),
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_plane_mid)
            .when(
                condition=(
                    spark_funcs.col("TRANSPORT") == "Plane") &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") > 3500),
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_plane_lng)
            .when(
                condition=(
                    spark_funcs.col("TRANSPORT") == "Train") &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") < 200),
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_train_shr)
            .when(
                condition=(
                    spark_funcs.col("TRANSPORT") == "Train") &
                    (spark_funcs.col("AJUSTED_DISTANCE_KM") >= 200),
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_train_tgv)
            .when(
                condition=spark_funcs.col("TRANSPORT") == "Taxi",
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_taxi)
            .when(
                condition=spark_funcs.col("TRANSPORT") == "Public transport",
                value=spark_funcs.col("AJUSTED_DISTANCE_KM") * ef_bus)
            .otherwise(None)) \
        .select(
            "KEY_MISSION",
            "KEY_PERSONNEL",
            "KEY_DATE",
            "KEY_DEPART",
            "TRANSPORT",
            "KEY_DESTINATION",
            "KEY_TYPE_MISSION",
            "EMISSIONS_T_CO2")

#────────────────────────────────────────────────────────────────────────────────────────────────────#

