import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LinearRegression
import time

def load_and_structure_data(file_path):
    # Adatok beolvasása CSV fájlból
    df = pd.read_csv(
       file_path, encoding="ISO-8859-2", sep=";"
    )  # sep paraméter a pontosvesszős fájlokhoz
    return df


# A fájl betöltése
file_path = r"data\stadat-nep0073-22.2.2.2-hu.csv"
df = load_and_structure_data(file_path)

# Strukturált adatok mentése új fájlba (például JSON formátumban)
df.to_json(
    "strukturalt_adatok.json", orient="records", force_ascii=False, indent=4
)  # A JSON jól olvasható lesz

df = pd.read_json("strukturalt_adatok.json")

# Vonaldiagram - Népesség alakulása negyedév szerint - Németh Péter

# JSON adatok betöltése fájlból
with open('strukturalt_adatok.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Vonaldiagram készítése minden területi egységhez
plt.figure(figsize=(15, 8))

for entry in json_data:
    quarters = list(entry.keys())[2:]  # Az első két kulcs kihagyása
    values = [int(entry[quarter].replace(" ", "")) if entry[quarter] is not None else None for quarter in quarters]
    plt.plot(quarters, values, marker='o', linestyle='-', label=entry["Területi egység neve"])

plt.xticks(rotation=45)
plt.xlabel('Negyedéves gördülő időszak')
plt.ylabel('Élve születések száma')
plt.title('Negyedéves adatok különböző területi egységek számára - gördülő időszakban')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) #Jelmagyarázat elhelyezése a diagramon kívül
plt.grid(True)
plt.tight_layout()

# Diagram mentése
plt.savefig('Elve_szuletesek_line_chart.png')
plt.show()


# Lineáris regresszió
# Adatok átalakítása DataFrame-be
data_list = []
for entry in json_data:
    quarters = list(entry.keys())[2:]  # Az első két kulcs kihagyása
    values = [
        int(entry[quarter].replace(" ", "")) if entry[quarter] is not None else None
        for quarter in quarters
    ]
    for i, quarter in enumerate(quarters):
        if values[i] is not None:
            data_list.append(
                {
                    "Területi egység neve": entry["Területi egység neve"],
                    "Negyedév": quarter,
                    "Élve születések száma": values[i],
                }
            )

df = pd.DataFrame(data_list)

# Az adatok csoportosítása területi egységek szerint (összegzett értékek számítása)
grouped_data = (
    df.groupby("Területi egység neve")["Élve születések száma"].sum().reset_index()
)

# Numerikus indexek a területi egységekhez
grouped_data["Területi index"] = pd.factorize(grouped_data["Területi egység neve"])[0]

# Lineáris regresszió számítása
x = grouped_data["Területi index"].values.reshape(-1, 1)  # X tengely: numerikus index
y = grouped_data["Élve születések száma"].values  # Y tengely: születésszámok
regressor = LinearRegression()
regressor.fit(x, y)

# Eredmények
slope = regressor.coef_[0]
intercept = regressor.intercept_
print(f"Meredekség (slope): {slope}")
print(f"Tengelymetszet (intercept): {intercept}")

# Predikciós értékek (trendvonal)
grouped_data["Trendvonal"] = regressor.predict(x)

# Vizualizáció seaborn-nal
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.scatterplot(
    data=grouped_data,
    x="Területi egység neve",
    y="Élve születések száma",
    color="blue",
    s=100,
    label="Adatok",
)
sns.lineplot(
    data=grouped_data,
    x="Területi egység neve",
    y="Trendvonal",
    color="red",
    label="Trendvonal (lineáris regresszió)",
)

# Címek és tengelyek beállítása
plt.title("Lineáris regresszió - Területi egységek alapján", fontsize=16)
plt.xlabel("Területi egység neve", fontsize=12)
plt.ylabel("Összesített élve születések száma", fontsize=12)
plt.xticks(rotation=45)
plt.legend()

# PNG fájl mentése
plt.tight_layout()
plt.savefig("Linearis_regresszio_Teruleti_egysegek.png")
plt.show()
plt.close()

#  Plot-diagramot az adatok alapján - AdamF

def history_diagram(path):
    kapcs = False
    # csv fájl betöltése
    file_path = path
    data = pd.read_csv(file_path, encoding="latin1", delimiter=";")

    # Adat rendezés
    # Space törlése a megefelő formatum eléréshez
    numeric_columns = data.columns[
        2:
    ]  # Első 2 elemet kihagyjuk , így kapot értékekből kiszedjük a spaceket.
    data[numeric_columns] = (
        data[numeric_columns]
        .replace(" ", "", regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )

    # Itt kérjük be a felhasználótól hogy melyik régiónak szeretné megnézni a diagramait.
    while kapcs == False:
        region_name = input("Kérlek adj meg egy régió nevet: ")
        if region_name == "help":
            for i in data["Területi egység neve"]:
                print(i)

        elif region_name != "":
            for y in data["Területi egység neve"]:
                print(y)
                if y == region_name:
                    kapcs = True
        else:
            print("Kérlek használd a help parancsot hogy lássd az elérhető régiókat")

    # Szűrés a terület értékekre
    region_data = data[data["Területi egység neve"] == region_name]

    # Értékek kigyüjtése idő alapján.
    time_periods = numeric_columns
    values = region_data.iloc[0, 2:]

    # Diagram paraméterei (adatai)
    plt.figure(figsize=(14, 7))
    plt.bar(time_periods, values)
    # Diagram kinézetének beállítása .
    plt.title(f"Negyed éves kimutatások {region_name}", fontsize=16)
    plt.xlabel("Idő", fontsize=14)
    plt.ylabel("Értékek", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)

    # Itt el mentjük a diagramot a report mappába.
    clock = time.strftime("%e-%m-%Y-%H-%M")
    loc = "hist_" + str(region_name) + "_" + str(clock) + ".png"
    plt.tight_layout()
    plt.savefig(loc)
    print(f"Fájl elkészült a következő helyre: {loc}")


history_diagram(path="data/stadat-nep0073-22.2.2.2-hu.csv")
