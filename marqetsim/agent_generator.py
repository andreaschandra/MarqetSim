"""Function to geenerate agent."""

import random
import pandas as pd

# Full name
male_first = [
    "Adi",
    "Agus",
    "Ahmad",
    "Andi",
    "Arif",
    "Budi",
    "Dedi",
    "Eko",
    "Fajar",
    "Hadi",
    "Indra",
    "Joko",
    "Krisna",
    "Lukman",
    "Made",
    "Nanda",
    "Oki",
    "Putra",
    "Reza",
    "Sandi",
    "Tono",
    "Udin",
    "Vino",
    "Wawan",
    "Yanto",
    "Zaki",
    "Bayu",
    "Candra",
    "Dimas",
    "Edo",
    "Faisal",
    "Gilang",
    "Hanif",
    "Ilham",
    "Jaya",
    "Karyono",
    "Lutfi",
    "Mahendra",
    "Nugroho",
    "Oka",
    "Pramono",
    "Qomar",
    "Ridho",
    "Surya",
    "Taufik",
    "Umar",
    "Verry",
    "Widodo",
    "Yusuf",
    "Zulkifli",
    "Aditya",
    "Bambang",
    "Cahyo",
    "Dani",
    "Erlangga",
    "Fahmi",
    "Gunawan",
    "Hendra",
    "Irfan",
    "Jefri",
]

female_first = [
    "Ayu",
    "Bella",
    "Citra",
    "Dewi",
    "Eka",
    "Fitri",
    "Gita",
    "Hana",
    "Indah",
    "Jihan",
    "Kartika",
    "Lina",
    "Maya",
    "Novi",
    "Okta",
    "Putri",
    "Qori",
    "Rina",
    "Sari",
    "Tari",
    "Ulfa",
    "Vira",
    "Wulan",
    "Yuni",
    "Zahra",
    "Amelia",
    "Bunga",
    "Cika",
    "Dara",
    "Elsa",
    "Farah",
    "Gina",
    "Hesti",
    "Ika",
    "Jesi",
    "Kirana",
    "Lilis",
    "Mila",
    "Nisa",
    "Okti",
    "Prita",
    "Qiara",
    "Rara",
    "Siska",
    "Tina",
    "Uci",
    "Vina",
    "Winda",
    "Yuli",
    "Zara",
    "Anisa",
    "Bening",
    "Cindy",
    "Dinda",
    "Endah",
    "Fika",
    "Galuh",
    "Hilda",
    "Irma",
    "Jeni",
]

family_names = [
    "Wijaya",
    "Santoso",
    "Perdana",
    "Kusuma",
    "Pratama",
    "Nugraha",
    "Permana",
    "Sutanto",
    "Gunawan",
    "Setiawan",
    "Handoko",
    "Kurniawan",
    "Suryanto",
    "Hakim",
    "Rahman",
    "Suharto",
    "Wibowo",
    "Mahendra",
    "Anggara",
    "Saputra",
    "Saputri",
    "Hartono",
    "Lestari",
    "Andriani",
    "Purnomo",
    "Susanto",
    "Marlina",
    "Budiono",
    "Rahayu",
    "Iskandar",
    "Firmansyah",
    "Hermawan",
    "Sugiarto",
    "Wardani",
    "Oktaviani",
    "Nasution",
    "Siregar",
    "Hasibuan",
    "Tampubolon",
    "Siahaan",
    "Simanjuntak",
    "Lumban Gaol",
    "Manurung",
    "Silalahi",
    "Pardede",
    "Sitompul",
    "Pakpahan",
    "Turnip",
    "Panjaitan",
    "Sianturi",
    "Tarigan",
    "Saragih",
    "Sembiring",
    "Karo-Karo",
    "Ginting",
    "Perangin-angin",
    "Situmorang",
    "Simatupang",
    "Hutasoit",
    "Bangun",
]

# Define coherent geographical and demographic data
countries_data = {
    "Indonesia": {
        "cities": [
            "Jakarta",
            "Surabaya",
            "Bandung",
            "Bekasi",
            "Medan",
            "Depok",
            "Tangerang",
            "Palembang",
            "Semarang",
            "Makassar",
            "South Tangerang",
            "Batam",
            "Bandar Lampung",
            "Bogor",
            "Pekanbaru",
            "Padang",
            "Malang",
            "Denpasar",
            "Samarinda",
            "Tasikmalaya",
        ],
        "common_occupations": [
            "Software Engineer",
            "Teacher",
            "Accountant",
            "Doctor",
            "Business Analyst",
        ],
        "education_levels": ["Bachelor's Degree", "Master's Degree", "Diploma", "PhD"],
    },
    "Malaysia": {
        "cities": [
            "Kuala Lumpur",
            "Putrajaya",
            "Johor Bahru",
            "Penang",
            "Kota Kinabalu",
        ],
        "common_occupations": [
            "Software Engineer",
            "Teacher",
            "Accountant",
            "Doctor",
            "Business Analyst",
        ],
        "education_levels": [
            "Bachelor's Degree",
            "Master's Degree",
            "PhD",
            "High School",
            "Diploma",
        ],
    },
    "Singapore": {
        "cities": ["Singapore City", "Jurong", "Tampines", "Woodlands", "Bedok"],
        "common_occupations": [
            "Financial Analyst",
            "Software Engineer",
            "Marketing Manager",
            "Nurse",
            "Consultant",
        ],
        "education_levels": ["Bachelor's Degree", "Master's Degree", "Diploma", "PhD"],
    },
}

# Household types with typical sizes
household_types = {
    "Nuclear family": [3, 4, 5],
    "Single person": [1],
    "Couple": [2],
    "Extended family": [4, 5, 6, 7],
    "Shared housing": [2, 3, 4],
}


def generate_coherent_person():
    """Generate a coherent personal data record."""

    # Select country and corresponding data
    country = random.choice(list(countries_data.keys()))
    country_info = countries_data[country]

    # Generate basic demographics
    age = random.randint(22, 65)
    gender = random.choice(["Male", "Female"])

    # Generate full name
    if gender == "Male":
        full_name = random.choice(male_first) + " " + random.choice(family_names)
    else:
        full_name = random.choice(female_first) + " " + random.choice(family_names)

    # Education level influences occupation and age coherence
    education = random.choice(country_info["education_levels"])

    # Adjust age based on education (higher education = older minimum age)
    if education in ["Master's Degree", "PhD"]:
        age = max(age, 25)
    elif education == "Bachelor's Degree":
        age = max(age, 22)

    # Select city from the same country
    city = random.choice(country_info["cities"])

    # Nationality often matches country of residence (with some exceptions)
    if random.random() < 0.8:  # 80% chance nationality matches residence
        nationality = country
    else:
        nationality = random.choice(list(countries_data.keys()))

    # Select household type and corresponding size
    household_type = random.choice(list(household_types.keys()))
    household_size = random.choice(household_types[household_type])

    # Occupation influenced by education and age
    occupation = random.choice(country_info["common_occupations"])

    # Adjust occupation based on education
    if education == "High School" and occupation in ["Doctor", "Engineer"]:
        occupation = random.choice(
            ["Sales Assistant", "Customer Service", "Technician"],
        )
    elif education == "PhD" and occupation not in [
        "Doctor",
        "Engineer",
        "Software Engineer",
    ]:
        occupation = random.choice(
            ["Research Scientist", "Professor", "Senior Consultant"]
        )

    income_level = random.choice(["Low", "Medium", "High"])
    employment_status = random.choice(
        ["Employed", "Self-employed", "Student", "Retired", "Unemployed"]
    )

    marital_status = random.choice(
        ["Single", "Married", "Divorced", "Widowed", "In a relationship"]
    )

    return {
        "name": full_name,
        "age": age,
        "gender": gender,
        "nationality": nationality,
        "city_of_residence": city,
        "country_of_residence": country,
        "education": education,
        "income_level": income_level,
        "employment_status": employment_status,
        "occupation": occupation,
        "marital_status": marital_status,
        "household_size": household_size,
        "household_type": household_type,
    }


if __name__ == "__main__":

    # Generate 10 coherent personal data records
    data = []
    for i in range(10):
        person = generate_coherent_person()
        data.append(person)

    # Create DataFrame and display
    df = pd.DataFrame(data)
    print("Generated Personal Data (10 records):")
    print("=" * 50)
    for i, row in df.iterrows():
        print(f"\nPerson {i+1}:")
        for col, value in row.items():
            print(f"  {col.replace('_', ' ').title()}: {value}")

    # Display summary statistics
    print("\n" + "=" * 50)
    print("DATA DISTRIBUTION SUMMARY:")
    print("=" * 50)
    print(f"Countries: {df['country_residence'].value_counts().to_dict()}")
    print(f"Age range: {df['age'].min()} - {df['age'].max()}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"Education levels: {df['education'].value_counts().to_dict()}")
    print(f"Household types: {df['household_type'].value_counts().to_dict()}")
