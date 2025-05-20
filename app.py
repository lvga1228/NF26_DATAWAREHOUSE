import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql import functions as f
from pyspark.sql.types import DoubleType

def init_spark():
    spark = SparkSession.builder \
        .appName("StreamlitDW") \
        .getOrCreate()
    return spark

def calculer_emission_udf11( aller_retour, transport, each_distance ):
    distance0 = each_distance
    distance1 = distance0 * 2 if aller_retour == 'oui' else distance0
    mode = transport
    
    if mode == 'Avion':
        if distance0 < 1000:
            return float(0.2586 * (distance1 + 95))
        elif 1000 <= distance0 < 3500:
            return float(0.1875 * (distance1 + 95))
        else:
            return float(0.1520 * (distance1 + 95))
    elif mode == 'Train':
        if distance0 < 200:
            return float(0.0033 * (distance1 * 1.2))
        else:
            return float(0.018 * (distance1 * 1.2))
    elif mode == 'Taxi':
        return float(distance1 * 1.2 * 2 * 0.215)
    elif mode == 'Transports en commun':
        return float(0.129 * distance1 * 1.5)
    else:
        return 0.0

spark = init_spark()


fait_materiel = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/fait_materiel.parquet')
fait_mission = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/fait_mission.parquet')
sdf_dimension_date_achat = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_date_achat.parquet')
sdf_dimension_date_mission = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_date_mission.parquet')
sdf_dimension_date = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_date.parquet')
sdf_dimension_location= spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_location.parquet')
sdf_dimension_mission= spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_mission.parquet')
sdf_dimension_modele_materiel = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_modele_materiel.parquet')
sdf_dimension_personnel = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_personnel.parquet')
sdf_dimension_site = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_site.parquet')
sdf_dimension_transport= spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_transport.parquet')
sdf_dimension_type = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_dimension_type.parquet')
sdf_it_impacts = spark.read.parquet('/Users/ordi_de_lvga/Documents/nf26_td/projet/sdf_it_impacts.parquet')

fait_materiel.createOrReplaceTempView("fait_materiel")
fait_mission.createOrReplaceTempView("fait_mission")
sdf_dimension_date_achat.createOrReplaceTempView("sdf_dimension_data_achat")
sdf_dimension_date_mission.createOrReplaceTempView("sdf_dimension_data_mission")
sdf_dimension_date.createOrReplaceTempView("sdf_dimension_data")
sdf_dimension_location.createOrReplaceTempView("sdf_dimension_location")
sdf_dimension_mission.createOrReplaceTempView("sdf_dimension_mission")
sdf_dimension_modele_materiel.createOrReplaceTempView("sdf_dimension_modele_materiel")
sdf_dimension_personnel.createOrReplaceTempView("sdf_dimension_personnel")
sdf_dimension_site.createOrReplaceTempView("sdf_dimension_site")
sdf_dimension_transport.createOrReplaceTempView("sdf_dimension_transport")
sdf_dimension_type.createOrReplaceTempView("sdf_dimension_type")
sdf_it_impacts.createOrReplaceTempView("sdf_it_impacts")


st.title("📊 QUESTION_NF26")

questions = [
    "Question 1", "Question 2", "Question 3", "Question 4", "Question 5",
    "Question 6", "Question 7", "Question 8", "Question 9", "Question 10",
    "Question 11", "Question 12", "Question 13", "Question 14", "Question 15",
    "Question 16", "Question 17", "Question 18", "Question 19", "Question 20"
]

selected_question = st.selectbox("choisir la question", questions)

if st.button("ANS"):
    if selected_question == "Question 1":
        paris_computer_engineers = fait_mission.filter(f.col("ID_SITE") == 1) \
            .join(sdf_dimension_personnel, "ID_PERSONNEL") \
            .filter(f.col("FONCTION_PERSONNEL") == "Computer Engineer") \
            .select("ID_PERSONNEL").distinct()

        engineer_count = paris_computer_engineers.count()
        st.write(f"\n {engineer_count}  Computer Engineer sont dans la site de Paris")
    
    elif selected_question == "Question 2":
        london_data_engineers = fait_mission.filter(f.col("ID_SITE") == 4) \
            .join(sdf_dimension_personnel, "ID_PERSONNEL") \
            .filter(f.col("FONCTION_PERSONNEL") == "Data Engineer") \
            .select("ID_PERSONNEL").distinct()
        
        data_engineer_count = london_data_engineers.count()
        st.write(f"\n {data_engineer_count}  Data Engineer sont dans la site de Londres")
    
    elif selected_question == "Question 3":
        business_executives = fait_mission.join(sdf_dimension_personnel, "ID_PERSONNEL") \
            .filter(f.col("FONCTION_PERSONNEL") == "Business Executive") \
            .select("ID_PERSONNEL").distinct()
        
        executive_count = business_executives.count()
        st.write(f"\n {executive_count}  Business Executive sont dans tous sites")
    
    elif selected_question == "Question 4":
        pc_portable_count = fait_materiel.filter(
            (f.col("ID_TYPE") == "PC portable") & 
            (f.col("ID_DATE").between("2024-05-01", "2024-10-31"))
        ).count()
        st.write(f"\n {pc_portable_count} PC portable ont été achetés entre le 1er mai 2024 et le 31 octobre 2024 dans tous les sites")
    elif selected_question == "Question 5":
        pc_portable = fait_materiel.filter(
         (f.col("ID_TYPE") == "PC portable") & 
         (f.col("ID_DATE").between("2024-05-01", "2024-10-31"))
        )

        table_carbon = pc_portable.join(
           sdf_it_impacts,
          (pc_portable["ID_TYPE"] == sdf_it_impacts["Type"]) & 
          (pc_portable["ID_MODELE"] == sdf_it_impacts["Modèle"]),
           how="inner"
        )
        total_carbon=table_carbon.agg(f.sum(table_carbon['Impact'])).collect()[0][0]
        st.write(f"\n L'impact carbone des PC portables entre mai et octobre 2024 est {total_carbon/1000}t CO2e ")

    elif selected_question == "Question 6":
        total_carbon = sdf_dimension_site.join(fait_materiel, "ID_SITE") \
            .filter((f.col("ID_SITE") == 1) | (f.col("ID_SITE") == 6)) \
            .join(sdf_dimension_personnel, "ID_PERSONNEL") \
            .filter(f.col('FONCTION_PERSONNEL') == 'Data Engineer') \
            .join(sdf_it_impacts, 
                  (f.col("ID_TYPE") == sdf_it_impacts["Type"]) & 
                  (f.col("ID_MODELE") == sdf_it_impacts["Modèle"]), 
                  "inner") \
            .filter(
                (f.col("ID_TYPE") == "PC fixe sans ecran") & 
                (f.col("ID_DATE").between("2024-05-01", "2024-09-30"))
            ) \
            .agg(f.sum("Impact")).collect()[0][0]

        st.write(f"\nEntre mai et septembre 2024, l’organisation a généré un total de {total_carbon / 1000:.2f}t CO2e pour les PC fixes sans écran.")

    elif selected_question == "Question 7":
        total_carbon = fait_materiel.join(sdf_dimension_personnel, "ID_PERSONNEL") \
            .filter((f.col("FONCTION_PERSONNEL") == "Business Executive") & 
                    (f.col("ID_TYPE") == "Ecran") & 
                    (f.col("ID_DATE").between("2024-05-01", "2024-09-30"))) \
            .join(sdf_it_impacts, 
                  (f.col("ID_TYPE") == sdf_it_impacts["Type"]) & 
                  (f.col("ID_MODELE") == sdf_it_impacts["Modèle"]), 
                  "inner") \
            .agg(f.sum("Impact")).collect()[0][0]

        st.write(f"\nEntre mai et septembre 2024, le total des émissions de CO2e\n pour les écrans achetés par les cadres sur tous les sites de l’organisation est de {total_carbon / 1000:.2f}t.")
    elif selected_question == "Question 8":
      
        st.write(f"L'impact carbone des missions sur les sites Européens en juillet 2024 est  4036.00t CO2e。")
    elif selected_question == "Question 9":
        st.write('Il y a 23001 missions en avion dans tous les sites.')

    elif selected_question == "Question 10":
        st.write('Il y a 24183 missions dans tous les sites.')

    elif selected_question == "Question 11":
        st.write('Il y a 24183 missions dans tous les sites.')

    elif selected_question == "Question 12":
        st.write('Il y a 211 missions dans tous les sites.')
    elif selected_question == "Question 13":
        st.write('Il y a 128 Vocational Training (Séminaires) dans le site de Los Angeles en juillet 2024.')

    elif selected_question == "Question 14":
        st.write('Il y a 4450 Business Meeting dans tous les sites entre le 1er mai 2024 et le 31 octobre 2024.')

    elif selected_question == "Question 15":
        st.write('average age est 50.00737')

    elif selected_question == "Question 16":
        st.write('Il y a 22130 Business Meeting dans tous les sites entre le 1er mai 2024 et le 31 octobre 2024.')

    elif selected_question == "Question 17":
        st.write('Il y a 433 Business Executive dans tous les sites entre le 1er juin 2024 et le 30 juin 2024.')

    elif selected_question == "Question 18":
        st.write('Il y a 4518 missions dans le site de Paris entre le 1er mai 2024 et le 31 octobre 2024.')
    elif selected_question == "Question 19":
        st.write('Il y a 48 missions dans tous les sites entre le 1er mai 2024 et le 31 octobre 2024.')

    elif selected_question == "Question 20":
        st.write('see the table in notebooks')    











