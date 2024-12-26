import json
import time
import webbrowser
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import streamlit as st
from openai import OpenAI, OpenAIError
from pycaret.clustering import load_model, predict_model  # type: ignore

openai_client = OpenAI(api_key="openai_api_key")
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters
#
# Sidebar
#
with st.sidebar:
    st.markdown(("""
✍️ Wypełnij krótką ankietę, abyśmy mogli lepiej Cię poznać!  
🔍 Znajdziemy osoby o podobnych zainteresowaniach, które mogą stać się Twoimi nowymi znajomymi.
"""))
    age = st.selectbox("Wiek", ['', '<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['', 'Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['', 'Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['', 'Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.selectbox("Płeć", ['', 'Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

# Warunek sprawdzający, czy wszystkie pola zostały wypełnione
if all(person_df.iloc[0].dropna().values != ''):
    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()
 
    # Dodanie legendy do sidebaru
    with st.sidebar:
        st.markdown("### Legenda")  # Tytuł legendy
        st.markdown(
            """
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 20px; height: 20px; background-color: green; margin-right: 10px;"></div>
                <span>Twoja społeczność</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: gray; margin-right: 10px;"></div>
                <span>Pozostałe społeczności</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

#
# I cześć app - nazwa grupy, opis grupy, wykresy
#

    st.title(f'''Znajdź swoją społeczność dzięki AI: Twoja społeczność - {predicted_cluster_data['name']}''')
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

    col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
    with col2:
        st.markdown("##### Wielkość Twojej społeczności: ")
    with col3:
        st.metric("""  Liczba osób  """, len(same_cluster_df))

    # Mapowanie nazw do Twojej grupy i pozostałych
    all_df["Group"] = all_df["Cluster"] == predicted_cluster_id
    all_df["Group"] = all_df["Group"].replace({True: "Twoja grupa", False: "Pozostałe grupy"})

    group_counts = all_df["Group"].value_counts().reset_index()
    group_counts.columns = ["Group", "Count"]
    fig_additional = px.pie(
        group_counts,
        names="Group",
        values="Count",
        title="Udział osób z Twojej społeczności<br>na tle wszystkich osób <br> ze wszystkich społeczności<br>  ",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
    )
    fig_additional.update_layout(showlegend=False)  
    
    fig_age = px.histogram(
        all_df,
        x="age",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
        barmode="overlay",
    )
    fig_age.update_layout(
        title="Rozkład wieku osób z Twojej <br>społeczności na tle wszystkich",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
        showlegend=False,
    )

    fig_edu = px.histogram(
        all_df,
        x="edu_level",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
        barmode="overlay",
    )
    fig_edu.update_layout(
        title="Rozkład wykształcenie Twojej <br>społeczności na tle wszystkich",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
        showlegend=False,
    )

    fig_animals = px.histogram(
        all_df,
        x="fav_animals",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
        barmode="overlay",
    )
    fig_animals.update_layout(
        title="Rozkład ulubionych zwierząt Twojej <br>społeczności na tle wszystkich",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
        showlegend=False,
    )

    fig_place = px.histogram(
        all_df,
        x="fav_place",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
        barmode="overlay",
    )
    fig_place.update_layout(
        title="Rozkład ulubionych miejsc Twojej <br>społeczności na tle wszystkich",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
        showlegend=False,
    )

    fig_gender = px.histogram(
        all_df,
        x="gender",
        color="Group",
        color_discrete_map={"Twoja grupa": "green", "Pozostałe grupy": "gray"},
        barmode="overlay",
    )
    fig_gender.update_layout(
        title="Rozkład płci Twojej <br>społeczności na tle wszystkich",
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
        showlegend=False,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_additional, use_container_width=True)
    with col2:
        st.plotly_chart(fig_age, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_edu, use_container_width=True)
    with col2:
        st.plotly_chart(fig_gender, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_animals, use_container_width=True)
    with col2:
        st.plotly_chart(fig_place, use_container_width=True)

#
# II Część app - GPT - pomoc w organizacji planu spotkania na podstawie wcześniejszych danych i wprowadzonego miejsca spotkania
#

    st.title(":tada: Świetnie! Znalazłeś swoją grupę, która jest do Ciebie dopasowana :tada:")
    
    if st.button(f"Wejdź na grupę {predicted_cluster_data['name']} i poznaj pozostałych członków"):
        st.video("https://youtu.be/dQw4w9WgXcQ?si=PUhLQ2n0NRS8MZE6")
        time.sleep(1)
        st.toast("Never gonna give you up")
        time.sleep(1)
        st.toast("Never gonna let you down")
        time.sleep(1)
        st.toast("Never gonna run around and desert you")
        time.sleep(1)
        st.toast("Never gonna make you cry")
        time.sleep(1)
        st.toast("Never gonna say goodbye")
        time.sleep(1)
        st.toast( "Never gonna tell a lie and hurt you")
            
    st.markdown("""
                Chętnie Ci pomożemy w zorganizowaniu spotkania dla Twojej grupy w opraciu o Wasze preferencje uzyskane z ankiety.  
                Wystarczy, że przekażesz nam poniżej swój klucz API OpenAI i miejsce gdzie chciałbyś zorganizować spotkanie, a 
                o całą resztę nie musisz się martwić :wink: :muscle:""")

    # Inicjalizacja session_state dla "miejsce"
    if "miejsce" not in st.session_state:
        st.session_state.miejsce = ""

    # Pole tekstowe do wprowadzenia danych
    st.session_state.miejsce = st.text_input("Wprowadź miejsce spotkania:", st.session_state.miejsce)

    # Inicjalizacja klucza w session_state
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    user_key = st.text_input("Klucz API:", type="password", value=st.session_state["openai_api_key"])

    if "api_key_valid" not in st.session_state:
        st.session_state["api_key_valid"] = False  # Domyślnie klucz niezweryfikowany

    if user_key:
        try:
            # Ustaw klucz API w kliencie OpenAI
            openai_client.api_key = user_key
            # Testowe zapytanie z ChatCompletion
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """
                            Testowe zapytanie walidacyjne klucza API.
                        """
                    },
                    {"role": "user", "content": "Czy mój klucz API działa?"}
                ]
            )
            st.session_state["openai_api_key"] = user_key
            st.session_state["api_key_valid"] = True  # Klucz zweryfikowany
            st.success("Klucz API OpenAI wprowadzony :thumbsup:")
        except OpenAIError:
            st.session_state["api_key_valid"] = False  # Klucz nieważny
            st.error("Wprowadzony klucz API OpenAI jest nieprawidłowy. Sprawdź i spróbuj ponownie.")
        except Exception as e:
            st.session_state["api_key_valid"] = False  # Klucz nieważny
            st.error(f"Wystąpił nieoczekiwany błąd: {e}")

    # Warunek, aby aktywować przycisk tylko wtedy, gdy oba warunki są spełnione
    if st.button("Kliknij mnie po wypełnieniu :)", disabled=not st.session_state.miejsce or not st.session_state["api_key_valid"]):
        with st.spinner("Proszę czekać, przetwarzam dane..."):
            # Funkcja generująca odpowiedź chatbota
            def get_chatbot_reply(user_prompt):
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                                Wypowiadasz się w sposób humorystyczny, ale zawsze w sposób merytoryczny i w pełni profesjonalny.
                                Jeteś ekspertem z 15 letnim doświadczeniem w branży turystycznej.
                                Odpowiadaj zawsze w języku polskim.
                                Pomagasz w zorganizowaniu spotkania grupy, która w opraciu o takie same zainteresowania nawiązała znajomość. 
                                możesz się spotkać i proponowany plan spotkania. 
                                Jak starasz się być zabawny, to podkreślaj to emotikonami.
                                Proponowane miejsca mają być prawdziwe.
                            """
                        },
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                }

            # Stała treść zapytania
            static_prompt = (f"""
                Na początku przywitaj się i podziękuj za korzystanie z tej aplikacji.
                
                Dane na podstawie, których masz udzielić odpowiedz:
                Miejsce w którym dana osoba chciałaby zorganizować spotkanie dla grupy: {st.session_state.miejsce},
                Dane osoby która organizuje wyjazd: {person_df},
                Nazwa całej grupy do której została przypisana dana osoba {predicted_cluster_data['name']},
                Opis całej grupy do której została przypisana dana osoba {predicted_cluster_data['description']},
                
                Powiedz coś na temat grupy, czym się charakteryzuje, co lubi i jakie aktywności, miejsca mogą być idealne dla takich osób.

                Na podstawie miejsce w którym dana osoba chciałaby zorganizować spotkanie dla grupy({st.session_state.miejsce}) zaproponuj w najbliższej okolicy atrakcyjne 
                i nietypowe miejsce do spotkania.
                Jeżeli w najbliższej okolicy nie ma ciekawych atrakcji, to zaproponuj z wybranego miejsca przejazd wynajętym autokarem lub jeżeli daleko od polski to samolotem 
                w o wiele bardziej atrakcyjne miejsce.
                Przedstaw plan atrakcyjnych zajęć integracyjnych dla nowo poznanych osób oraz animacje które zapadną w pamięciu osobą dorosłym. 
                Zaproponuj plan na cały weekend piątek - niedziela, w planie zrób rozpiskę zajęć uwzględniającą cały dzień i wieczór.
                
                Wprowadź jedną atrakcją, która powiązana będzie z tym że wszystkie osoby na spotakaniu są z kursu "Pracuj w AI: Zostań Data Scientist od Zera". 
                Na sam koniec podziękuj jeszcze raz za skorzystanie z aplikacji i wyraź nadzieję że pomogłeś i zachęć do kontaktu z wymyśloną firmą turystyczną, 
                którą reprezentujesz(przedstaw wiytówkę firmy na koniec z pełnymi danymi kontaktkowymi - podanymi w tabeli oraz na sam koniec podpisz się śmiesznym imieniem nawiązującym do sztucznej inteligencji).
            """
            )

            chatbot_message = get_chatbot_reply(static_prompt)

            with st.chat_message("assistant"):
                st.markdown(chatbot_message["content"])

else:
    st.title("Znajdź swoją społeczność dzięki AI")
    st.markdown("""
    Nasza aplikacja wykorzystuje sztuczną inteligencję, aby łączyć użytkowników w grupy na podstawie wspólnych zainteresowań i cech.  
    🌟 Oferujemy personalizowane dopasowania, wizualizacje danych i pomoc w planowaniu wydarzeń!  
    👉 Wypełnij ankietę po lewej stronie, a resztą zajmie się AI.
                """)
    st.stop()